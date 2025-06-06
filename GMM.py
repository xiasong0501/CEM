import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class GMM(nn.Module):
    def __init__(self, num_components, input_dim):
        super(GMM, self).__init__()
        self.num_components = num_components
        self.input_dim = input_dim
        
        self.means = nn.Parameter(torch.randn(num_components, input_dim))
        self.covariances = nn.Parameter(torch.eye(input_dim).repeat(num_components, 1, 1))
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)
    
    def forward(self, x):
        batch_size = x.size(0)
        # print(batch_size)
        covariances = self.covariances + torch.eye(self.input_dim, device=x.device) * 1e-6
        
        # Compute the probability density for each component
        probs = torch.stack([
            MultivariateNormal(self.means[i], covariance_matrix=covariances[i]).log_prob(x)
            for i in range(self.num_components)
        ]).T
        
        # Multiply by the component weights
        weighted_probs = probs + torch.log(self.weights)
        
        # Logsumexp to get the log likelihood
        log_likelihood = torch.logsumexp(weighted_probs, dim=1)
        
        return log_likelihood
    
    def fit(self, x, num_epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = -self.forward(x).mean()
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    def get_params(self):
        return {
            'weights': self.weights.detach().cpu().numpy(),
            'means': self.means.detach().cpu().numpy(),
            'covariances': self.covariances.detach().cpu().numpy()
        }

def fit_gmm_torch(features, labels, num_classes, num_gaussians, device='cuda'):
    features = features.view(features.size(0), -1).to(device)
    labels = labels.to(device)
    
    gmm_params = {}
    
    for cls in range(num_classes):
        cls_features = features[labels == cls]
        gmm = GMM(num_gaussians, cls_features.size(1)).to(device)
        gmm.fit(cls_features)
        
        gmm_params[cls] = gmm.get_params()
    
    return gmm_params
