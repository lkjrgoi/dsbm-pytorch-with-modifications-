import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключает предупреждения TensorFlow
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
import copy
import ot as pot
try:
    import torchdiffeq
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Neural ODE functionality will be limited.")

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
dataset_size = 10000 #Размер обучающего набора
test_dataset_size = 10000# Размер тестового набора
lr = 1e-4
batch_size = 128


class MLP(nn.Module):#стандартный многослойный перцептрон
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super(MLP, self).__init__()
    layers = []
    prev_width = input_dim
    for layer_width in layer_widths:
      layers.append(torch.nn.Linear(prev_width, layer_width))
      prev_width = layer_width
    self.input_dim = input_dim
    self.layer_widths = layer_widths
    self.layers = nn.ModuleList(layers)
    self.activate_final = activate_final
    self.activation_fn = activation_fn
        
  def forward(self, x):
    for i, layer in enumerate(self.layers[:-1]):
      x = self.activation_fn(layer(x))
    x = self.layers[-1](x)
    if self.activate_final:
      x = self.activation_fn(x)
    return x


class ScoreNetwork(nn.Module):#сеть для оценки градиента
  def __init__(self, input_dim, layer_widths=[100,100,2], activate_final = False, activation_fn=F.tanh):
    super().__init__()
    self.net = MLP(input_dim, layer_widths=layer_widths, activate_final=activate_final, activation_fn=activation_fn)
    """Прямой проход с конкатенацией входа и времени"""
  def forward(self, x_input, t):
    inputs = torch.cat([x_input, t], dim=1)# Объединение данных и времени
    return self.net(inputs)

# Класс Diffusion Schrödinger Bridge (DSB)
# Original DSB
class DSB(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=20, sig=0):
    """
        :param net_fwd: сеть для прямого процесса
        :param net_bwd: сеть для обратного процесса
        :param num_steps: количество шагов дискретизации
        :param sig: уровень шума (σ) - параметр стохастичности
    """
    super().__init__()
    self.net_fwd = net_fwd# сеть для прямого направления
    self.net_bwd = net_bwd# сеть для обратного направления
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}# Словарь сетей
    self.N = num_steps# количество шагов дискретизации
    self.sig = sig# параметр диффузии
  
  @torch.no_grad()
  def generate_new_dataset_and_train_tuple(self, x_pairs=None, fb='', first_it=False):# Генерация новых данных для обучения, генерирует новые данные для обучения, симулируя траектории
    """
        Генерация нового датасета и обучающих данных
        :param x_pairs: пары точек (начальные и конечные состояния)
        :param fb: направление ('f' - forward, 'b' - backward)
        :param first_it: флаг первой итерации
    """
    assert fb in ['f', 'b']
# Выбор начальной точки в зависимости от направления
    if fb == 'f':
      prev_fb = 'b'
      zstart = x_pairs[:, 1]# начинаем с конечного распределения, берем конечные точки
    else:
      prev_fb = 'f'
      zstart = x_pairs[:, 0]# начинаем с начального распределения, берем начальные точки
    N = self.N
    dt = 1./N# Размер шага по времени
    traj = [] # to store the trajectory
    signal = [] #Для хранения целевых значений
    tlist = []# Для хранения временных меток
    z = zstart.detach().clone()
    batchsize = zstart.shape[0]
    dim = zstart.shape[1]
    # Временные точки
    ts = np.arange(N) / N
    tl = np.arange(1, N+1) / N
    if prev_fb == 'b':
      ts = 1 - ts# Обратное время для обратного процесса
      tl = 1 - tl
    # Генерация траектории  
    if first_it:#first_it - флаг первой итерации (использует простой диффузионный процесс), на первой итерации просто добавляем шум
      assert prev_fb == 'f'
      for i in range(N):
        t = torch.ones((batchsize,1), device=device) * ts[i]
        dz = self.sig * torch.randn_like(z) * np.sqrt(dt)# Добавление стохастичности: σ * dW, где dW ~ N(0, sqrt(dt))
        z = z + dz
        tlist.append(torch.ones((batchsize,1), device=device) * tl[i])
        traj.append(z.detach().clone())
        signal.append(-dz)# Целевое значение - отрицательный шум
    else:
      # На последующих итерациях используем обученную сеть
      for i in range(N):
        t = torch.ones((batchsize,1), device=device) * ts[i]
        pred = self.net_dict[prev_fb](z, t)
        z = z.detach().clone() + pred
        dz = self.sig * torch.randn_like(z) * np.sqrt(dt)# Добавление стохастичности
        z = z + dz
        tlist.append(torch.ones((batchsize,1), device=device) * tl[i])
        traj.append(z.detach().clone())
        signal.append(- self.net_dict[prev_fb](z, t) - dz)# Комбинированное целевое значение
    # Упаковка данных
    z_t = torch.stack(traj)
    tlist = torch.stack(tlist)
    target = torch.stack(signal)
    # Случайный выбор точек из траектории
    randint = torch.randint(N, (1, batchsize, 1), device=device)
    tlist = torch.gather(tlist, 0, randint).squeeze(0)
    z_t = torch.gather(z_t, 0, randint.expand(1, batchsize, dim)).squeeze(0)
    target = torch.gather(target, 0, randint.expand(1, batchsize, dim)).squeeze(0)
    return z_t, tlist, target

  @torch.no_grad()
  def sample_sde(self, zstart=None, fb='', first_it=False, N=None):
    """Выборка из обученного диффузионного процесса"""
    assert fb in ['f', 'b']
    # Сэмплирование из обученной модели с помощью численного решения SDE
    ### NOTE: Use Euler method to sample from the learned flow
    N = self.N
    dt = 1./N
    traj = [] # to store the trajectory, для хранения траектории
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts# обратное время для backward процесса
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)# Предсказание сети
      z = z.detach().clone() + pred
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)# Добавление стохастичности: σ * dW
      traj.append(z.detach().clone())
    return traj


def train_dsb_ipf(dsb_ipf, x_pairs, batch_size, inner_iters, fb='', first_it=False, **kwargs):# Обучение DSB методом IPF (Iterative Proportional Fitting)
  """
    Обучает модель DSB (прямую или обратную сеть) используя метод итеративного пропорционального фиттинга (IPF)
    
    Параметры:
        dsb_ipf: объект DSB (содержит net_fwd и net_bwd)
        x_pairs: тензор пар точек [batch_size, 2, dim] - начальные и конечные точки траекторий
        batch_size: размер батча для обучения
        inner_iters: количество внутренних итераций обучения
        fb: направление обучения ('f' - forward, 'b' - backward)
        first_it: флаг первой итерации (используется для особой инициализации)
        **kwargs: дополнительные аргументы
  """
  #Генерирует данные, создает DataLoader и оптимизирует параметры сети
  # Проверка что направление задано корректно
  assert fb in ['f', 'b']
  dsb_ipf.fb = fb# Установка направления обучения в объекте DSB
  optimizer = torch.optim.Adam(dsb_ipf.net_dict[fb].parameters(), lr=lr)# Создание оптимизатора для соответствующей сети (прямой или обратной)
  # optimizer = dsb_ipf.optimizer_dict[fb]
  loss_curve = []
  # Генерация нового обучающего набора:
  # z_ts - точки траектории, ts - временные метки, targets - целевые значения
  z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)# Генерация данных для обучения
  dl = iter(DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))# Создание DataLoader для итерации по батчам

  for i in tqdm(range(inner_iters)):# Цикл обучения на inner_iters итерациях
    try:# Попытка взять следующий батч
      z_t, t, target = next(dl)
    except StopIteration:# Если данные закончились, генерируем новый набор
      z_ts, ts, targets = dsb_ipf.generate_new_dataset_and_train_tuple(x_pairs=x_pairs, fb=fb, first_it=first_it)
      dl = iter(DataLoader(TensorDataset(z_ts, ts, targets), batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z_t, t, target = next(dl)
    
    optimizer.zero_grad()
    pred = dsb_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return dsb_ipf, loss_curve


# DSBM
'''
Ключевые отличия DSBM от других методов
Чередование направлений: DSBM попеременно обучает прямой и обратный процессы, что улучшает стабильность обучения
Метод Matching: В отличие от DSB, который использует score matching, DSBM непосредственно обучает сети предсказывать оптимальные переходы
Гибкость начального сопряжения: DSBM поддерживает разные стратегии начального сопряжения ("ref" и "ind"), что позволяет лучше контролировать процесс обучения
Эффективность: DSBM требует меньше памяти, так как не нужно сохранять все промежуточные точки траекторий.
'''
class DSBM(nn.Module):
  def __init__(self, net_fwd=None, net_bwd=None, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref"):
    super().__init__()
    self.net_fwd = net_fwd
    self.net_bwd = net_bwd
    self.net_dict = {"f": self.net_fwd, "b": self.net_bwd}
    # self.optimizer_dict = {"f": torch.optim.Adam(self.net_fwd.parameters(), lr=lr), "b": torch.optim.Adam(self.net_bwd.parameters(), lr=lr)}
    self.N = num_steps
    self.sig = sig
    self.eps = eps#eps - малый параметр, предотвращающий деление на 0 при t близких к 0 или 1
    self.first_coupling = first_coupling#first_coupling - стратегия начального сопряжения ("ref" - референсное, "ind" - независимое)
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', **kwargs):#Генерирует данные для обучения в соответствии с методом Matching
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps# линейная интерполяция
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z# добавление шума, z_t - интерполированная точка между z0 и z1 с добавленным шумом
    if fb == 'f':
      # z1 - z_t / (1-t)
      target = z1 - z0 #целевое значение для обучения сети (различается для прямого и обратного направлений)
      target = target - self.sig * torch.sqrt(t/(1.-t)) * z
    else:
      # z0 - z_t / t
      target = - (z1 - z0)
      target = target - self.sig * torch.sqrt((1.-t)/t) * z
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    assert fb in ['f', 'b']                                                       
    '''
    Генерирует новые пары (z0, z1) для обучения
    На первой итерации использует либо референсное, либо независимое сопряжение
    На последующих итерациях использует предыдущую модель для генерации конечных точек
    '''                                                                              
    if prev_model is None:
      assert first_it
      assert fb == 'b'# Инициализация для первой итерации
      # Первая итерация
      zstart = x_pairs[:, 0]
      if self.first_coupling == "ref":
        # First coupling is x_0, x_0 perturbed
        zend = zstart + torch.randn_like(zstart) * self.sig
      elif self.first_coupling == "ind":
        zend = x_pairs[:, 1].clone()
        zend = zend[torch.randperm(len(zend))]# перестановка для независимого сопряжения
      else:
        raise NotImplementedError
      z0, z1 = zstart, zend
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:# fb == 'f'
        zstart = x_pairs[:, 1]
      # Последующие итерации - используем предыдущую модель для генерации
      zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N   
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = self.net_dict[fb](z, t)
      z = z.detach().clone() + pred * dt
      z = z + self.sig * torch.randn_like(z) * np.sqrt(dt)
      traj.append(z.detach().clone())

    return traj

#Обучение DSBM с чередованием прямого и обратного направлений
#Использует предыдущую модель для генерации более качественных данных
def train_dsbm(dsbm_ipf, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
  assert fb in ['f', 'b']
  dsbm_ipf.fb = fb
  optimizer = torch.optim.Adam(dsbm_ipf.net_dict[fb].parameters(), lr=lr)# Обучение DSBM
  # optimizer = dsbm_ipf.optimizer_dict[fb]
  loss_curve = []
  
  dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))# Генерация новых данных с учетом предыдущей модели

  for i in tqdm(range(inner_iters)):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*dsbm_ipf.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)
    
    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = dsbm_ipf.get_train_tuple(z_pairs, fb=fb, first_it=first_it)
    optimizer.zero_grad()
    pred = dsbm_ipf.net_dict[fb](z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return dsbm_ipf, loss_curve

class ODEFunc(nn.Module):
    """Neural ODE функция с поддержкой батчового времени для моста Шредингера"""
    def __init__(self, input_dim, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim + 1  # +1 для времени
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
    
    def forward(self, t, x):
        """
        определяет правую часть нашего дифференциального уравнения: dx/dt = f(t, x)
        t: scalar или tensor [batch_size] (время для каждой точки)
        x: tensor [batch_size, input_dim] (состояния)
        """
        # Проверка входной размерности
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")
        
        # Обработка времени
        if isinstance(t, torch.Tensor) and t.dim() == 1:
            # Батч времени: t shape [batch_size] -> [batch_size, 1]
            t_vector = t.unsqueeze(1)
        elif isinstance(t, torch.Tensor) and t.dim() == 0:
            # Скалярное время в тензоре
            t_vector = t.item() * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        else:
            # Скалярное время (float)
            t_vector = t * torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        # Время t конкатенируется с состоянием x, чтобы сеть могла обучаться зависимости производной от времени
        # Конкатенация: [batch_size, input_dim] + [batch_size, 1] = [batch_size, input_dim+1]
        inputs = torch.cat([x, t_vector], dim=1)
        
        return self.net(inputs)


class DSBM_NeuralODE(nn.Module):
    def __init__(self, input_dim, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref", traj_file=None, pretrain_epochs=100):
        super().__init__()
        self.net_forward = ODEFunc(input_dim).to(device)  # Используем ODEFunc для прямой сети и обратной 
        self.net_backward = ODEFunc(input_dim).to(device)
        self.optimizer_forward = torch.optim.Adam(self.net_forward.parameters(), lr=lr)
        self.optimizer_backward = torch.optim.Adam(self.net_backward.parameters(), lr=lr)
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        self.first_coupling = first_coupling
        self.traj_file = traj_file
        self.input_dim = input_dim
        self.trained_forward = False
        self.trained_backward = False
        self.fb = None
        self.pretrain_epochs = pretrain_epochs

        if traj_file:
          self._load_and_preprocess_trajectories()
        else:
            self.traj_tensor = None
            self.dataset = None
            self.dataloader = None

    def _load_and_preprocess_trajectories(self):
        """Загрузка и подготовка данных траекторий"""
        try:
          traj_data = np.load(self.traj_file)
          print(f"Loaded trajectory data with shape: {traj_data.shape}")
        
          self.traj_tensor = torch.tensor(traj_data, dtype=torch.float32, device=device)
        
          # Подготовка пар (z_t, z_{t+1}) траекторий для обучения
          self.X = self.traj_tensor[:, :-1, :].reshape(-1, self.traj_tensor.shape[-1])
          self.y = self.traj_tensor[:, 1:, :].reshape(-1, self.traj_tensor.shape[-1])
        
          self.dataset = TensorDataset(self.X, self.y)
          self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, 
                                   pin_memory=False)
        except FileNotFoundError:
            print(f"Trajectory file {self.traj_file} not found, skipping pretraining")
            self.traj_tensor = None
            self.dataset = None
            self.dataloader = None

    def train_on_trajectories(self, direction='both',epochs=None):
        """Обучение NeuralODE на загруженных траекториях"""
        if self.traj_tensor is None or self.dataloader is None:
            print("No trajectory data available for pretraining")
            return
        
        if epochs is None:
            epochs = self.pretrain_epochs  # Используем pretrain_epochs, если явно не указано

        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss_forward = 0
            epoch_loss_backward = 0
            batch_count = 0

            for batch_X, batch_y in self.dataloader:
                batch_count += 1
                # Обучение forward сети
                if direction in ['both', 'forward']:
                    self.optimizer_forward.zero_grad()
                    t = torch.rand(1, device=device).item()# Скаляр времени, для каждого батча используем случайное время (выбирается случайным образом)
                    pred_dx = self.net_forward(t, batch_X)# Предсказание производной
                    pred_y = batch_X + pred_dx * (1.0/self.N)  # шаг Эйлера
                    loss_forward = criterion(pred_y, batch_y)
                    loss_forward.backward()
                    self.optimizer_forward.step()
                    epoch_loss_forward += loss_forward.item()
                
                # Обучение backward сети  
                if direction in ['both', 'backward']:
                    self.optimizer_backward.zero_grad()
                    t = torch.rand(1, device=device).item()  # Скаляр времени
                    pred_dx = self.net_backward(t, batch_y)  # Предсказание производной
                    pred_x = batch_y + pred_dx * (1.0/self.N)  # шаг Эйлера
                    loss_backward = criterion(pred_x, batch_X)
                    loss_backward.backward()
                    self.optimizer_backward.step()
                    epoch_loss_backward += loss_backward.item()
            
            # Вычисление среднего лосса
            if batch_count > 0:
                if direction in ['both', 'forward']:
                    avg_loss_forward = epoch_loss_forward / batch_count
                    print(f"Pretrain Epoch {epoch}, Forward Loss: {avg_loss_forward:.6f}")
            
                if direction in ['both', 'backward']:
                    avg_loss_backward = epoch_loss_backward / batch_count
                    print(f"Pretrain Epoch {epoch}, Backward Loss: {avg_loss_backward:.6f}")
    
        print(f"Pretraining completed for {direction} direction")
    
    def get_network(self, direction):
        """Возвращает соответствующую сеть для направления"""
        if direction == 'f':
            return self.net_forward
        elif direction == 'b':
            return self.net_backward
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def get_optimizer(self, direction):
        """Возвращает соответствующий оптимизатор"""
        if direction == 'f':
            return self.optimizer_forward
        elif direction == 'b':
            return self.optimizer_backward
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
        """Генерация обучающих данных"""
        z0, z1 = x_pairs[:, 0], x_pairs[:, 1]
        
        t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
        z_t = t * z1 + (1.-t) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
        
        # Для Neural ODE цель - производная (скорость изменения)
        if fb == 'f':
            target = z1 - z0 - self.sig * torch.sqrt(t/(1.-t)) * z
        else:
            target = -(z1 - z0) - self.sig * torch.sqrt((1.-t)/t) * z
        
        return z_t, t, target

    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
        """Генерация новых пар (z0, z1) для обучения"""
        assert fb in ['f', 'b']
        
        if prev_model is None:
            assert first_it
            assert fb == 'b'
            zstart = x_pairs[:, 0]
            if self.first_coupling == "ref":
                zend = zstart + torch.randn_like(zstart) * self.sig
            elif self.first_coupling == "ind":
                zend = x_pairs[:, 1].clone()
                zend = zend[torch.randperm(len(zend))]
            else:
                raise NotImplementedError
            z0, z1 = zstart, zend
        else:
            assert not first_it
            if prev_model.fb == 'f':
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            
            # Используем sample_sde для генерации конечных точек
            zend = prev_model.sample_sde(zstart=zstart, fb=prev_model.fb)[-1]
            
            if prev_model.fb == 'f':
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        
        return z0, z1
    
    @torch.no_grad()
    def sample_sde(self, zstart=None, N=None, fb='', first_it=False):
        """Семплирование SDE версии (Эйлер-Маруяма)"""
        assert fb in ['f', 'b']
        
        if N is None:
            N = self.N
        
        traj = [zstart.detach().clone()]
        z = zstart.detach().clone()
        dt = 1.0 / N
        
        net = self.get_network(fb)
        
        if fb == 'f':
            time_points = torch.linspace(0, 1, N, device=device)
        else:
            time_points = torch.linspace(1, 0, N, device=device)
        
        for i in range(N):
            t = time_points[i].item()  # Скаляр времени
            # Детерминированный шаг
            dz_det = net(t, z) * dt
            # Стохастический шаг
            
            dz_stoch = self.sig * torch.randn_like(z) * math.sqrt(dt)
            
            z = z + dz_det + dz_stoch
            traj.append(z.detach().clone())
        
        return traj
    
    @torch.no_grad()
    def sample_ode(self, zstart=None, N=None, fb=''):
        """Семплирование ODE версии (если доступен torchdiffeq)"""
        assert fb in ['f', 'b']
        
        if not TORCHDIFFEQ_AVAILABLE:
            print("Warning: torchdiffeq not available, using Euler method")
            return self.sample_sde(zstart, N, fb)
        
        if N is None:
            N = self.N
        
        net = self.get_network(fb)
        
        if fb == 'f':
            t_span = torch.linspace(0, 1, N, device=device)
        else:
            t_span = torch.linspace(1, 0, N, device=device)
        
        # Решаем ODE
        traj = torchdiffeq.odeint(net, zstart, t_span, method='euler')
        
        return traj

def train_dsbm_neuralode(dsbm_model, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
    """Функция обучения DSBM с NeuralODE для задачи Шредингера"""
    assert fb in ['f', 'b']
    dsbm_model.fb = fb
    optimizer = dsbm_model.get_optimizer(fb)
    loss_curve = []
    
    # Претренинг если нужно (только на первой итерации)
    if first_it and not (dsbm_model.trained_forward if fb == 'f' else dsbm_model.trained_backward):
        print(f"Initial training on saved trajectories for {fb} direction...")
        dsbm_model.train_on_trajectories(direction='forward' if fb == 'f' else 'backward', 
                                       epochs=min(50, dsbm_model.pretrain_epochs))
    
    # Генерация нового датасета в соответствии с процедурой IPF
    z0, z1 = dsbm_model.generate_new_dataset(x_pairs, prev_model, fb, first_it)
    z_pairs = torch.stack([z0, z1], dim=1)
    
    # Создание обучающих данных (аналогично DSBM)
    z_t, t, target = dsbm_model.get_train_tuple(z_pairs, fb=fb)
    
    # Проверка размерностей
    print(f"Training data shapes - z_t: {z_t.shape}, t: {t.shape}, target: {target.shape}")
    print(f"Network expects input dimension: {dsbm_model.net_forward.input_dim}")
    
    dataset = TensorDataset(z_t, t, target)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=False)
    
    early_stop = False
    
    for i in tqdm(range(inner_iters), desc=f"Training {fb} direction"):
        if early_stop:
            break
            
        epoch_loss = 0
        num_batches = 0
        
        for z_t_batch, t_batch, target_batch in dl:
            optimizer.zero_grad()
            net = dsbm_model.get_network(fb)
            
            # КОРРЕКТНО: передаем время для КАЖДОЙ точки индивидуально
            # t_batch: [batch_size, 1] -> [batch_size]
            t_batch = t_batch.squeeze(1)
            
            # Важная проверка размерностей
            if z_t_batch.shape[1] != dsbm_model.input_dim:
                raise ValueError(f"Batch dimension mismatch: expected {dsbm_model.input_dim}, got {z_t_batch.shape[1]}")
            
            # Каждая точка обучается при СВОЕМ времени - это важно для Шредингера!
            pred = net(t_batch, z_t_batch)
            
            # MSE loss между предсказанной и целевой производной
            loss = F.mse_loss(pred, target_batch)
            loss.backward()
            
            # Градиентный clipping для стабильности
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            loss_curve.append(loss.item())
            
            if torch.isnan(loss).any():
                print("NaN loss detected, stopping training")
                early_stop = True
                break
        
        # Логирование прогресса
        if num_batches > 0 and i % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Iteration {i}, Average Loss: {avg_loss:.6f}")
            
            # Ранняя остановка если loss не улучшается
            if i > 100 and avg_loss > 10.0:  # Эвристика для обнаружения расходимости
                print("Loss too high, stopping training")
                early_stop = True
                break
    
    # Обновляем флаги обучения после завершения
    if fb == 'f':
        dsbm_model.trained_forward = True
        print("Forward network training completed")
    else:
        dsbm_model.trained_backward = True
        print("Backward network training completed")
    
    return dsbm_model, loss_curve

# SB-CFM, Schrodinger Bridge with Conditional Flow Matching
#Использует семплер оптимального транспорта (Sinkhorn) для получения начального сопряжения

class SBCFM(nn.Module):
  def __init__(self, net=None, num_steps=1000, sig=0, eps=1e-3):
    super().__init__()
    self.net = net
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
    self.N = num_steps
    self.sig = sig
    self.eps = eps
    from bridge.sde.optimal_transport import OTPlanSampler
    self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * sig**2)#reg - параметр регуляризации для Sinkhorn алгоритма
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, **kwargs):
    x0, x1 = x_pairs[:, 0], x_pairs[:, 1]
    z0, z1 = self.ot_sampler.sample_plan(x0, x1)

    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    z = torch.randn_like(z_t)
    z_t = z_t + self.sig * torch.sqrt(t*(1.-t)) * z
    target = z1 - z0 
    target = target - self.sig * (torch.sqrt(t)/torch.sqrt(1.-t) - 0.5 / torch.sqrt(t*(1.-t))) * z
    return z_t, t, target
    
  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, **kwargs):
    return x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]

  @torch.no_grad()
  def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    sign = 1 if fb == 'f' else -1
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = sign * self.net(z, t)
      z = z.detach().clone() + pred * dt
      traj.append(z.detach().clone())

    return traj


# Rectified Flow, в отличие от DSBM, не использует диффузионный член (чистый ODE)
class RectifiedFlow(nn.Module):
  def __init__(self, net=None, num_steps=1000, sig=0, eps=0):
    super().__init__()
    self.net = net
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)  # torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
    self.N = num_steps
    self.sig = sig# не используется, так как Rectified Flow детерминированный
    self.eps = eps
  
  @torch.no_grad()
  def get_train_tuple(self, x_pairs=None, fb='', first_it=False):
    z0, z1 = x_pairs[:, 0], x_pairs[:, 1]

    t = torch.rand((z1.shape[0], 1), device=device) * (1-2*self.eps) + self.eps
    z_t = t * z1 + (1.-t) * z0
    target = z1 - z0
    return z_t, t, target

  @torch.no_grad()
  def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
    if prev_model is None:
      assert first_it
      z0, z1 = x_pairs[:, 0], x_pairs[torch.randperm(len(x_pairs)), 1]
    else:
      assert not first_it
      if prev_model.fb == 'f':
        zstart = x_pairs[:, 0]
      else:
        zstart = x_pairs[:, 1]
      zend = prev_model.sample_ode(zstart=zstart, fb=prev_model.fb)[-1]
      if prev_model.fb == 'f':
        z0, z1 = zstart, zend
      else:
        z0, z1 = zend, zstart
    return z0, z1

  @torch.no_grad()
  def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
    assert fb in ['f', 'b']
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = zstart.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    ts = np.arange(N) / N
    if fb == 'b':
      ts = 1 - ts
    sign = 1 if fb == 'f' else -1
    for i in range(N):
      t = torch.ones((batchsize,1), device=device) * ts[i]
      pred = sign * self.net(z, t)
      z = z.detach().clone() + pred * dt
      traj.append(z.detach().clone())

    return traj


def train_flow_model(flow_model, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
  assert fb in ['f', 'b']
  flow_model.fb = fb
  optimizer = flow_model.optimizer
  loss_curve = []
  
  dl = iter(DataLoader(TensorDataset(*flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                       batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))

  for i in tqdm(range(inner_iters)):
    try:
      z0, z1 = next(dl)
    except StopIteration:
      dl = iter(DataLoader(TensorDataset(*flow_model.generate_new_dataset(x_pairs, prev_model=prev_model, fb=fb, first_it=first_it)), 
                           batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True))
      z0, z1 = next(dl)

    z_pairs = torch.stack([z0, z1], dim=1)
    z_t, t, target = flow_model.get_train_tuple(x_pairs=z_pairs, fb=fb, first_it=first_it)

    optimizer.zero_grad()
    pred = flow_model.net(z_t, t)
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()
    
    if torch.isnan(loss).any():
      raise ValueError("Loss is nan")
      break
    
    optimizer.step()
    loss_curve.append(np.log(loss.item())) ## to store the loss curve

  return flow_model, loss_curve


@torch.no_grad()
def draw_plot(sample_fn, z0, z1, N=None):
  traj = sample_fn(N=N)
  
  plt.figure(figsize=(4,4))
  plt.xlim(-5,5)
  plt.ylim(-5,5)
    
  plt.scatter(z0[:, 0].cpu().numpy(), z0[:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
  plt.legend()
  plt.title('Distribution')
  plt.tight_layout()

  # traj_particles = torch.stack(traj)
  # plt.figure(figsize=(4,4))
  # plt.xlim(-5,5)
  # plt.ylim(-5,5)
  # plt.axis('equal')
  # for i in range(30):
  #   plt.plot(traj_particles[:, i, 0].cpu(), traj_particles[:, i, 1].cpu())
  # plt.title('Transport Trajectory')
  # plt.tight_layout()


def train(cfg: DictConfig):#Основной тренировочный цикл
  # set seed for random number generators in pytorch, numpy and python.random
  if cfg.get("seed"):
    print(f"Seed: <{cfg.seed}>")
    pl.seed_everything(cfg.seed, workers=True)

  a = cfg.a
  dim = cfg.dim
  # Инициализация данных
  initial_model = Normal(-a * torch.ones((dim, )), 1)#Инициализация двух гауссовских распределений с разными средними
  target_model = Normal(a * torch.ones((dim, )), 1)
  
  x0 = initial_model.sample([dataset_size])#Создание пар данных (x0, x1) для обучения
  x1 = target_model.sample([dataset_size])
  x_pairs = torch.stack([x0, x1], dim=1).to(device)
  
  x0_test = initial_model.sample([test_dataset_size])
  x1_test = target_model.sample([test_dataset_size])
  x0_test = x0_test.to(device)
  x1_test = x1_test.to(device)

  torch.save({'x0': x0, 'x1': x1, 'x0_test': x0_test, 'x1_test': x1_test}, "data.pt")

  x_test_dict = {'f': x0_test, 'b': x1_test}
  
  net_split = cfg.net_name.split("_")
  if net_split[0] == "mlp":
    if net_split[1] == "small":
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[128, 128, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
    else:
      net_fn = partial(ScoreNetwork, input_dim=dim+1, layer_widths=[256, 256, dim], activation_fn=hydra.utils.get_class(cfg.activation_fn)())  # hydra.utils.get_method(cfg.activation_fn))  # 
  else:
    raise NotImplementedError
  
  num_steps = cfg.num_steps
  sigma = cfg.sigma
  inner_iters = cfg.inner_iters
  outer_iters = cfg.outer_iters
# Создание модели в зависимости от конфигурации (DSB, DSBM, SB-CFM или Rectified Flow) и выбор соответствующей функции обучения
  if cfg.model_name == "dsb":
    model = DSB(net_fwd=net_fn().to(device), 
                net_bwd=net_fn().to(device), 
                num_steps=num_steps, sig=sigma)
    train_fn = train_dsb_ipf
    print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "dsbm":
    model = DSBM(net_fwd=net_fn().to(device), 
                  net_bwd=net_fn().to(device), 
                  num_steps=num_steps, sig=sigma, first_coupling=cfg.first_coupling)
    train_fn = train_dsbm
    print(f"Number of parameters: <{sum(p.numel() for p in model.net_fwd.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "dsbm_neuralode":
    model = DSBM_NeuralODE(input_dim=dim, 
                          num_steps=num_steps,
                          sig=sigma,
                          first_coupling=cfg.first_coupling,
                          traj_file='/home/user1/dsbm-pytorch/traj.npy',
                          pretrain_epochs=100)
    train_fn = train_dsbm_neuralode
    print("Using DSBM with Neural ODE")
  
  elif cfg.model_name == "sbcfm":
    model = SBCFM(net=net_fn().to(device), 
                  num_steps=num_steps, sig=sigma)
    train_fn = train_flow_model
    print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
  elif cfg.model_name == "rectifiedflow":
    model = RectifiedFlow(net=net_fn().to(device), 
                          num_steps=num_steps, sig=None)
    train_fn = train_flow_model
    print(f"Number of parameters: <{sum(p.numel() for p in model.net.parameters() if p.requires_grad)}>")
  else:
    raise ValueError("Wrong model_name!")


  # Training loop
  # first_it = True
  model_list = []
  it = 1

  # Основной цикл обучения с чередованием направлений, после каждой итерации сохраняет модель и визуализирует результаты
  # assert outer_iters % len(cfg.fb_sequence) == 0
  while it <= outer_iters:
    for fb in cfg.fb_sequence:
      print(f"Iteration {it}/{outer_iters} {fb}")
      first_it = (it == 1)
      if first_it:
        prev_model = None
      else:
        prev_model = model_list[-1]["model"].eval()
      model, loss_curve = train_fn(model, x_pairs, batch_size, inner_iters, prev_model=prev_model, fb=fb, first_it=first_it)
      model_list.append({'fb': fb, 'model': copy.deepcopy(model).eval()})
    # Визуализация и оценка
      if hasattr(model, "sample_sde"):
        draw_plot(partial(model.sample_sde, zstart=x_test_dict[fb], fb=fb, first_it=first_it), z0=x_test_dict['f'], z1=x_test_dict['b'])
        plt.savefig(f"{it}-{fb}.png")
        plt.close()

        # Evaluation
        optimal_result_dict = {'mean': -a, 'var': 1, 'cov': (np.sqrt(5) - 1) / 2}
        result_list = {k: [] for k in optimal_result_dict.keys()}
        for i in range(len(model_list)):
          try:
            traj = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b')
            result_list['mean'].append(traj[-1].mean(0).mean(0).item())
            result_list['var'].append(traj[-1].var(0).mean(0).item())
            result_list['cov'].append(torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
          except Exception as e:
            print(f"Error processing model {i}: {e}")
            # Добавляем NaN для сохранения одинаковой длины
            result_list['mean'].append(float('nan'))
            result_list['var'].append(float('nan'))
            result_list['cov'].append(float('nan'))    
                
        for i, k in enumerate(result_list.keys()):
          plt.plot(result_list[k], label=f"{cfg.model_name}-{cfg.net_name}")
          plt.plot(np.arange(outer_iters), optimal_result_dict[k] * np.ones(outer_iters), label="optimal", linestyle="--")
          plt.title(k.capitalize())
          if i == 0:
            plt.legend()
          plt.savefig(f"convergence_{k}.png")
          plt.close()
        
        result_list_100 = {k: [] for k in optimal_result_dict.keys()}
        for i in range(it):
          traj_100 = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b', N=100)
          result_list_100['mean'].append(traj_100[-1].mean(0).mean(0).item())
          result_list_100['var'].append(traj_100[-1].var(0).mean(0).item())
          result_list_100['cov'].append(torch.cov(torch.cat([traj_100[0], traj_100[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
          
      if hasattr(model, "sample_ode"):
        draw_plot(partial(model.sample_ode, zstart=x_test_dict[fb], fb=fb), z0=x_test_dict['f'], z1=x_test_dict['b'])
        plt.savefig(f"{it}-{fb}-ode.png")
        plt.close()

        # Evaluation
        optimal_result_dict_ode = {'mean': -a, 'var': 1, 'cov': (np.sqrt(5) - 1) / 2}#
        result_list_ode = {k: [] for k in optimal_result_dict_ode.keys()}
        for i in range(it):
          traj_ode = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b')
          result_list_ode['mean'].append(traj_ode[-1].mean(0).mean(0).item())
          result_list_ode['var'].append(traj_ode[-1].var(0).mean(0).item())
          result_list['cov'].append(torch.cov(torch.cat([traj_ode[0], traj_ode[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())#
        for i, k in enumerate(result_list_ode.keys()):
          plt.plot(result_list_ode[k], label=f"{cfg.model_name}-{cfg.net_name}-ode")
          plt.plot(np.arange(outer_iters), optimal_result_dict_ode[k] * np.ones(outer_iters), label="optimal", linestyle="--")
          plt.title(k.capitalize())
          if i == 0:
            plt.legend()
          plt.savefig(f"convergence_{k}-ode.png")
          plt.close()
        
        result_list_ode_100 = {k: [] for k in optimal_result_dict_ode.keys()}
        for i in range(len(model_list)):
          try:
            traj_ode_100 = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b', N=100)
            result_list_ode_100['mean'].append(traj_ode_100[-1].mean(0).mean(0).item())
            result_list_ode_100['var'].append(traj_ode_100[-1].var(0).mean(0).item())
            result_list_ode_100['cov'].append(torch.cov(torch.cat([traj_ode_100[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())#
          except Exception as e:
            print(f"Error processing model {i}: {e}")
            # Добавляем NaN для сохранения одинаковой длины
            result_list['mean'].append(float('nan'))
            result_list['var'].append(float('nan'))
            result_list['cov'].append(float('nan'))
      # first_it = False
      it += 1

      if it > outer_iters:
        break

  torch.save([{'fb': m['fb'], 'model': m['model'].state_dict()} for m in model_list], "model_list.pt")

  if hasattr(model, "sample_sde"):
    df_result = pd.DataFrame(result_list)
    df_result_100 = pd.DataFrame(result_list_100)
    df_result.to_csv('df_result.csv')
    df_result.to_pickle('df_result.pkl')
    df_result_100.to_csv('df_result_100.csv')
    df_result_100.to_pickle('df_result_100.pkl')

    # Trajectory
    np.save("traj.npy", torch.stack(traj, dim=1).detach().cpu().numpy())
    np.save("traj_100.npy", torch.stack(traj_100, dim=1).detach().cpu().numpy())

  if hasattr(model, "sample_ode"):
    df_result_ode = pd.DataFrame(result_list_ode)
    df_result_ode_100 = pd.DataFrame(result_list_ode_100)
    df_result_ode.to_csv('df_result_ode.csv')
    df_result_ode.to_pickle('df_result_ode.pkl')
    df_result_ode_100.to_csv('df_result_ode_100.csv')
    df_result_ode_100.to_pickle('df_result_ode_100.pkl')

    # Trajectory
    np.save("traj_ode.npy", torch.stack(traj_ode, dim=1).detach().cpu().numpy())
    np.save("traj_ode_100.npy", torch.stack(traj_ode_100, dim=1).detach().cpu().numpy())

  return {}, {}


@hydra.main(config_path="conf", config_name="gaussian.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)

if __name__ == "__main__":
    main()