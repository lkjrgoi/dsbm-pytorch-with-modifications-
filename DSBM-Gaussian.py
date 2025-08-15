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

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig


device = 'cuda'
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
    """Новый класс для NeuralODE, заменяющий ScoreNetwork"""
    def __init__(self, input_dim, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x):
        return self.net(x)


class DSBM_NeuralODE(nn.Module):
    def __init__(self, input_dim, num_steps=1000, sig=0, eps=1e-3, first_coupling="ref", traj_file='/home/user1/dsbm-pytorch/traj.npy', pretrain_epochs=1000):
        super().__init__()
        self.net = ODEFunc(input_dim).to(device)  # Используем новый класс ODEFunc
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.N = num_steps
        self.sig = sig
        self.eps = eps
        self.first_coupling = first_coupling
        self.traj_file = traj_file
        self.trained = False
        self.fb = None
        self.pretrain_epochs = pretrain_epochs
        self._load_and_preprocess_trajectories()
    
    def _load_and_preprocess_trajectories(self):
        """Загрузка и подготовка данных траекторий"""
        traj_data = np.load(self.traj_file)
        print(f"Loaded trajectory data with shape: {traj_data.shape}")
        
        self.traj_tensor = torch.tensor(traj_data, dtype=torch.float32, device=device)
        
        # Подготовка пар (z_t, z_{t+1}) для обучения
        self.X = self.traj_tensor[:, :-1, :].reshape(-1, self.traj_tensor.shape[-1])
        self.y = self.traj_tensor[:, 1:, :].reshape(-1, self.traj_tensor.shape[-1])
        
        self.dataset = TensorDataset(self.X, self.y)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def train_on_trajectories(self, epochs=1000):
        """Обучение NeuralODE на загруженных траекториях"""
        if self.trained:
            return
        if epochs is None:
            epochs = self.pretrain_epochs  # Используем pretrain_epochs, если явно не указано

        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in self.dataloader:
                self.optimizer.zero_grad()
                
                # Предсказываем следующую точку
                pred_y = batch_X + self.net(0, batch_X) * (1.0/self.N)
                
                loss = criterion(pred_y, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss/len(self.dataloader)}")
        
        self.trained = True
    
    @torch.no_grad()
    def get_train_tuple(self, x_pairs=None, fb='', **kwargs):
        """Генерация обучающих данных"""
        if not self.trained:
            self.train_on_trajectories()
        
        idx = torch.randint(0, len(self.dataset), (1,)).item()
        z_t = self.dataset[idx][0].unsqueeze(0)
        t = torch.rand((1, 1), device=device) * (1-2*self.eps) + self.eps
        target = self.dataset[idx][1].unsqueeze(0) - z_t
        
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, zstart=None, N=None, fb='', first_it=False):
        """Семплирование траектории"""
        if N is None:
            N = self.N
        
        traj = [zstart.detach().clone()]
        z = zstart.detach().clone()
        dt = 1.0 / N
        sign = 1 if fb == 'f' else -1
        
        for _ in range(N):
            z = z + sign * self.net(0, z) * dt
            traj.append(z.detach().clone())
        
        return traj
    @torch.no_grad()
    def generate_new_dataset(self, x_pairs, prev_model=None, fb='', first_it=False):
        """Генерация новых пар (z0, z1) для обучения"""
        assert fb in ['f', 'b']
        
        if prev_model is None:
            assert first_it
            assert fb == 'b'
            # Первая итерация
            zstart = x_pairs[:, 0]
            if self.first_coupling == "ref":
                # First coupling is x_0, x_0 perturbed
                zend = zstart + torch.randn_like(zstart) * self.sig
            elif self.first_coupling == "ind":
                zend = x_pairs[:, 1].clone()
                zend = zend[torch.randperm(len(zend))]  # перестановка для независимого сопряжения
            else:
                raise NotImplementedError
            z0, z1 = zstart, zend
        else:
            assert not first_it
            if prev_model.fb == 'f':
                zstart = x_pairs[:, 0]
            else:
                zstart = x_pairs[:, 1]
            # Последующие итерации - используем предыдущую модель для генерации
            zend = prev_model.sample_ode(zstart=zstart, fb=prev_model.fb)[-1]
            if prev_model.fb == 'f':
                z0, z1 = zstart, zend
            else:
                z0, z1 = zend, zstart
        return z0, z1

def train_dsbm_neuralode(dsbm_model, x_pairs, batch_size, inner_iters, prev_model=None, fb='', first_it=False):
    """Функция обучения DSBM с NeuralODE"""
    if first_it and not dsbm_model.trained:
        print("Initial training on saved trajectories...")
        dsbm_model.train_on_trajectories(epochs=None)
        return dsbm_model, []
    
    dsbm_model.fb = fb
    optimizer = dsbm_model.optimizer
    loss_curve = []
    
    dl = iter(DataLoader(TensorDataset(*dsbm_model.generate_new_dataset(x_pairs, prev_model, fb, first_it)), 
                     batch_size=batch_size, shuffle=True))
    
    for i in tqdm(range(inner_iters)):
        try:
            z0, z1 = next(dl)
        except StopIteration:
            dl = iter(DataLoader(TensorDataset(*dsbm_model.generate_new_dataset(x_pairs, prev_model, fb, first_it)), 
                         batch_size=batch_size, shuffle=True))
            z0, z1 = next(dl)
        
        z_pairs = torch.stack([z0, z1], dim=1)
        z_t, t, target = dsbm_model.get_train_tuple(z_pairs, fb=fb)
        
        optimizer.zero_grad()
        pred = dsbm_model.net(0, z_t)  # Время не используется в нашей NeuralODE
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
    
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
                          pretrain_epochs=1000)
    train_fn = train_dsbm_neuralode
  
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
        for i in range(it):
          traj = model_list[i]['model'].sample_sde(zstart=x1_test, fb='b')
          result_list['mean'].append(traj[-1].mean(0).mean(0).item())
          result_list['var'].append(traj[-1].var(0).mean(0).item())
          result_list['cov'].append(torch.cov(torch.cat([traj[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())
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
        draw_plot(partial(model.sample_ode, zstart=x_test_dict[fb], fb=fb, first_it=first_it), z0=x_test_dict['f'], z1=x_test_dict['b'])
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
        for i in range(it):
          traj_ode_100 = model_list[i]['model'].sample_ode(zstart=x1_test, fb='b', N=100)
          result_list_ode_100['mean'].append(traj_ode_100[-1].mean(0).mean(0).item())
          result_list_ode_100['var'].append(traj_ode_100[-1].var(0).mean(0).item())
          result_list_ode_100['cov'].append(torch.cov(torch.cat([traj_ode_100[0], traj[-1]], dim=1).T)[dim:, :dim].diag().mean(0).item())#
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