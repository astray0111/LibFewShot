import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model.finetuning.finetuning_model import FinetuningModel
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from transformers import CLIPTextModel
from core.utils import accuracy
import clip

class FDAlign(FinetuningModel):
    """Feature Discrimination Alignment for Fine-tuning Pre-trained Models"""
    
    def __init__(self, feat_dim, num_class=64, alpha=1.0, beta=20.0, **kwargs):
        super(FDAlign, self).__init__(**kwargs)
        
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.alpha = alpha
        self.beta = beta
        
        # Initialize CLIP
        self.clip_model, _ = clip.load("ViT-B/32", device='cpu')
        
        # Freeze text encoder
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # 训练阶段使用 num_class，测试阶段使用 way_num
        self.train_classifier = nn.Linear(feat_dim, num_class)
        
        # Cache for validation phase
        self.cached_prototypes = None
        
        # Default templates
        self.templates = [
            "a photo of a {}",
            "a photograph of a {}",
            "an image of a {}",
        ]
        
    def _get_text_features(self, class_names):
        """Extract text features using templates"""
        text_features = []
        for class_name in class_names:
            # Generate text descriptions using templates
            class_texts = [template.format(class_name) for template in self.templates]
            
            # Encode text using CLIP
            with torch.no_grad():
                text_tokens = clip.tokenize(class_texts).to(self.device)
                text_feats = self.clip_model.encode_text(text_tokens)
                text_features.append(text_feats)
            
        return torch.stack(text_features)  # [n_classes, n_templates, feat_dim]
        
    def _optimize_templates(self, text_features):
        """Remove outlier templates and cluster similar ones"""
        # Get shapes
        n_classes, n_templates, feat_dim = text_features.shape
        
        # Move features to CPU for sklearn operations
        text_features_reshaped = text_features.view(-1, feat_dim).cpu()
        
        # Detect outliers using Isolation Forest
        iso = IsolationForest(contamination=0.25)  # 保留约60%的模板
        valid_mask = iso.fit_predict(text_features_reshaped)
        valid_mask = valid_mask.reshape(n_classes, n_templates)
        
        # Collect valid features
        valid_features = []
        for i in range(n_classes):
            class_valid_feats = text_features[i, valid_mask[i] == 1]
            if len(class_valid_feats) > 0:
                valid_features.append(class_valid_feats)
            else:
                # If no valid templates, keep all templates for this class
                valid_features.append(text_features[i])
        
        # Ensure all classes have same number of templates
        min_templates = min(feat.size(0) for feat in valid_features)
        valid_features = [feat[:min_templates] for feat in valid_features]
        valid_features = torch.stack(valid_features)
        
        # Convert features for K-means
        valid_features_reshaped = valid_features.view(-1, feat_dim).cpu()
        
        # Cluster similar templates
        # 确保聚类数至少为1，最多为有效样本数
        n_clusters = max(1, min(
            valid_features.size(1),  # 每个类的模板数
            n_classes * 2,  # 类别数的两倍
            3  # 最少3个聚类
        ))
        
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(valid_features_reshaped)
        clusters = clusters.reshape(n_classes, -1)
        
        # Average features within each cluster
        optimized_features = []
        for i in range(n_classes):
            class_features = []
            for j in range(n_clusters):
                cluster_mask = clusters[i] == j
                if cluster_mask.any():
                    cluster_mean = valid_features[i, cluster_mask].mean(0)
                else:
                    # 如果某个类别缺少某个聚类，使用该类别的平均特征
                    cluster_mean = valid_features[i].mean(0)
                class_features.append(cluster_mean)
            optimized_features.append(torch.stack(class_features))
        
        return torch.stack(optimized_features)  # [n_classes, n_clusters, feat_dim]
        
    def _compute_kl_div(self, p, q):
        """Compute KL divergence between two distributions"""
        return F.kl_div(F.log_softmax(p, dim=-1), 
                       F.softmax(q, dim=-1), 
                       reduction='batchmean')
        
    def set_forward(self, batch):
        """Inference phase"""
        images, _ = batch
        images = images.to(self.device)
        
        # Extract features
        feat = self.emb_func(images)
        
        # Split features into support and query
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        
        # Only compute prototypes if not cached
        if self.cached_prototypes is None:
            self.cached_prototypes = self.get_prototypes(support_feat[0], support_target[0])
        
        # Get predictions for query set
        logits = self.get_score(query_feat.reshape(-1, self.feat_dim), self.cached_prototypes)
        
        # Compute accuracy
        acc = accuracy(logits, query_target.reshape(-1))
        
        return logits, acc
        
    def get_prototypes(self, support_feat, support_target):
        """Compute class prototypes from support features"""
        # Generate pseudo class names
        unique_labels = support_target.unique().tolist()
        class_names = [f"class_{label}" for label in unique_labels]
        
        # Extract text features
        text_features = self._get_text_features(class_names)
        
        # Optimize templates
        optimized_features = self._optimize_templates(text_features)
        
        # Compute prototypes
        prototypes = optimized_features.mean(1)  # [n_classes, feat_dim]
        
        return prototypes
        
    def get_score(self, query_feat, prototypes):
        """Compute classification scores"""
        # Normalize features
        query_feat = F.normalize(query_feat, p=2, dim=-1)
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        # Compute cosine similarity
        logits = torch.mm(query_feat, prototypes.t())
        
        return logits 

    def train(self, mode=True):
        """Override train method to handle cache"""
        super().train(mode)
        if mode:  # training mode
            self.cached_prototypes = None
        return self

    def eval(self):
        """Override eval method to handle cache"""
        super().eval()
        self.cached_prototypes = None
        return self 

    def set_forward_loss(self, batch):
        """Training phase"""
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Extract features using current encoder
        feat = self.emb_func(images)  # [batch_size, feat_dim]
        
        # Extract features using original CLIP (detached)
        with torch.no_grad():
            orig_feat = self.clip_model.encode_image(images)
        
        # Generate pseudo class names for current batch
        unique_labels = targets.unique().tolist()
        class_names = [f"class_{label}" for label in unique_labels]
        
        # Get text features and optimize templates
        text_features = self._get_text_features(class_names)
        optimized_features = self._optimize_templates(text_features)
        
        # Get spurious prototypes
        spurious_prototypes = optimized_features.mean(0)  # [n_templates, feat_dim]
        
        # Compute distributions over spurious prototypes
        feat_norm = F.normalize(feat, p=2, dim=-1)
        orig_feat_norm = F.normalize(orig_feat, p=2, dim=-1)
        spurious_prototypes_norm = F.normalize(spurious_prototypes, p=2, dim=-1)
        
        # Current distribution
        curr_dist = torch.matmul(feat_norm, spurious_prototypes_norm.t())
        # Original distribution
        orig_dist = torch.matmul(orig_feat_norm, spurious_prototypes_norm.t())
        
        # Compute KL divergence loss
        kl_loss = self._compute_kl_div(curr_dist, orig_dist)
        
        # Classification loss
        logits = self.train_classifier(feat)
        cls_loss = F.cross_entropy(logits, targets)
        
        # Total loss
        loss = self.alpha * cls_loss + self.beta * kl_loss
        
        # Compute accuracy
        acc = accuracy(logits, targets)
        
        return logits, acc, loss 