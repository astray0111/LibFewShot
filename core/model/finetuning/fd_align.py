import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.finetuning.finetuning_model import FinetuningModel
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np
import copy

class FDAlign(FinetuningModel):
    def __init__(self, feat_dim, num_class, inner_param, **kwargs):
        super(FDAlign, self).__init__(**kwargs)
        
        # 基础分类器
        self.classifier = nn.Linear(feat_dim, num_class)
        self.loss_func = nn.CrossEntropyLoss()
        
        # FD-Align 参数
        self.inner_param = inner_param
        self.temp = inner_param.get('temperature', 0.1)
        self.alpha = 1.0  # 分类损失权重
        self.beta = 20.0  # 特征对齐损失权重
        
        # SPC 参数
        self.n_templates = inner_param.get('n_templates', 60)  # 保留的模板数
        self.n_clusters = inner_param.get('n_clusters', 20)   # 聚类中心数
        
        # 保存原始backbone
        self.emb_func_orig = None
        
        # 计数器
        self.episode_count = 0
        
        # 初始化网络
        self._init_network()
        
    def _init_network(self):
        """初始化网络"""
        super()._init_network()
        # 确保 emb_func 已经初始化
        if self.emb_func is not None:
            # 深度复制原始backbone
            self.emb_func_orig = copy.deepcopy(self.emb_func)
            self.emb_func_orig.eval()
            for param in self.emb_func_orig.parameters():
                param.requires_grad = False
            
    def forward_output(self, x):
        feat = self.emb_func(x)
        out = self.classifier(feat)
        return out, feat
        
    def set_forward(self, batch):
        """推理阶段"""
        image, global_target = batch
        image = image.to(self.device)
        
        # 获取输入数据的形状
        b, c, h, w = image.shape
        total_samples = self.way_num * (self.shot_num + self.query_num)
        
        # 分割 support 和 query 集
        support_size = self.way_num * self.shot_num
        query_size = self.way_num * self.query_num
        
        support_image = image[:support_size]
        query_image = image[support_size:total_samples]
        
        # 生成标签
        support_target = torch.arange(self.way_num).repeat_interleave(self.shot_num).to(self.device)
        query_target = torch.arange(self.way_num).repeat_interleave(self.query_num).to(self.device)
        
        # 微调阶段
        self.set_forward_adaptation(support_image, support_target)
        
        # 预测
        output, _ = self.forward_output(query_image)
        acc = self.accuracy(output, query_target)
        
        return output, acc
        
    def set_forward_loss(self, batch):
        """训练阶段"""
        # 更新并打印进度
        self.episode_count += 1
        if self.episode_count % 100 == 0:  # 每100个episode打印一次
            print(f"Episode: {self.episode_count}")
            
        image, global_target = batch
        image = image.to(self.device)
        
        # 获取输入数据的形状
        b, c, h, w = image.shape
        total_samples = self.way_num * (self.shot_num + self.query_num)
        
        # 分割 support 和 query 集
        support_size = self.way_num * self.shot_num
        query_size = self.way_num * self.query_num
        
        support_image = image[:support_size]
        query_image = image[support_size:total_samples]
        
        # 生成标签
        support_target = torch.arange(self.way_num).repeat_interleave(self.shot_num).to(self.device)
        query_target = torch.arange(self.way_num).repeat_interleave(self.query_num).to(self.device)
        
        # 微调阶段
        self.train()
        optimizer = self.sub_optimizer(self.classifier, {
            'name': 'SGD',
            'kwargs': {
                'lr': self.inner_param.get('lr', 0.01),
                'momentum': 0.9,
                'weight_decay': 0.0005
            }
        })
        
        for _ in range(self.inner_param.get('adaptation_steps', 100)):
            output, feat = self.forward_output(support_image)
            loss = self.loss_func(output, support_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 前向传播
        output, feat = self.forward_output(query_image)
        
        # 确保 emb_func_orig 已初始化
        if self.emb_func_orig is None:
            self._init_network()
            
        with torch.no_grad():
            feat_orig = self.emb_func_orig(query_image)  # 原始特征
        
        # 1. 分类损失
        cls_loss = self.loss_func(output, query_target)
        
        # 2. 特征对齐损失
        # 计算特征相似度
        feat_norm = F.normalize(feat, dim=1)
        feat_orig_norm = F.normalize(feat_orig, dim=1)
        
        # 计算特征分布
        sim_matrix = torch.matmul(feat_norm, feat_orig_norm.t()) / self.temp
        
        # 计算正样本对的相似度
        pos_mask = (query_target.unsqueeze(0) == query_target.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # 排除自身
        pos_sim = (sim_matrix * pos_mask).sum(1) / (pos_mask.sum(1) + 1e-8)
        
        # 计算负样本对的相似度
        neg_mask = 1 - pos_mask
        neg_sim = (sim_matrix * neg_mask).sum(1) / (neg_mask.sum(1) + 1e-8)
        
        # 对比损失
        align_loss = torch.mean(torch.relu(neg_sim - pos_sim + 0.1))
            
        # 总损失
        loss = self.alpha * cls_loss + self.beta * align_loss
        acc = self.accuracy(output, query_target)
        
        return output, acc, loss
        
    def set_forward_adaptation(self, support_image, support_target):
        """微调阶段"""
        # 创建优化器配置
        optim_config = {
            'name': 'SGD',
            'kwargs': {
                'lr': self.inner_param.get('lr', 0.01),
                'momentum': 0.9,
                'weight_decay': 0.0005
            }
        }
        
        optimizer = self.sub_optimizer(self.classifier, optim_config)
        
        self.train()
        for _ in range(self.inner_param.get('adaptation_steps', 100)):
            output, feat = self.forward_output(support_image)
            loss = self.loss_func(output, support_target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def accuracy(self, output, target):
        """计算准确率"""
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item()