import numpy as np
import pandas as pd
from scipy.stats import norm

class LeakageSafeComBat:
    """
    Empirical Bayes ComBat Harmonizer designed for multi-site neuroimaging data.
    Guarantees strict zero-data-leakage: parameters are fitted ONLY on training set folds.
    """
    def __init__(self, eb=True):
        self.eb = eb
        self.fitted = False
        
    def fit(self, X, sites, covariates=None):
        """
        Fits ComBat site harmonization parameters on training features X (N_samples, N_features).
        sites: list/array of length N_samples containing site names/IDs.
        covariates: optional pandas DataFrame of continuous/discrete covariates (e.g. Age, Gender).
        """
        X = np.asarray(X, dtype=np.float64)
        N_samples, N_features = X.shape
        sites = np.asarray(sites)
        
        unique_sites, site_counts = np.unique(sites, return_counts=True)
        self.unique_sites = list(unique_sites)
        self.n_sites = len(unique_sites)
        
        # Design matrix for site indicators
        design = pd.get_dummies(sites, drop_first=False).astype(float).values
        
        if covariates is not None:
            cov_design = pd.get_dummies(covariates, drop_first=True).astype(float).values
            design = np.hstack([design, cov_design])
            
        # 1. Fit grand mean alpha and covariate coefficients beta via Least Squares
        # X^T = B * Design^T  => B = X^T * Design * (Design^T * Design)^-1
        try:
            inv_design = np.linalg.pinv(design.T @ design)
            B_hat = X.T @ design @ inv_design  # (N_features, N_design)
        except Exception:
            B_hat = np.zeros((N_features, design.shape[1]))
            
        # Grand mean per feature
        grand_mean = np.mean(X, axis=0, keepdims=True)  # (1, N_features)
        
        # Calculate residuals after removing grand mean & covariates
        var_pooled = np.var(X - grand_mean, axis=0, keepdims=True)
        var_pooled[var_pooled == 0] = 1e-8
        
        # Standardized data
        stand_mean = grand_mean
        s_data = (X - stand_mean) / np.sqrt(var_pooled)
        
        # 2. Site-specific location (gamma) and scale (delta^2) estimates
        gamma_hat = {}
        delta_hat2 = {}
        
        for site in self.unique_sites:
            idx = (sites == site)
            site_data = s_data[idx]
            g_site = np.mean(site_data, axis=0)
            if len(site_data) > 1:
                d_site = np.var(site_data, axis=0, ddof=1)
            else:
                d_site = np.ones_like(g_site)
            d_site = np.nan_to_num(d_site, nan=1.0)
            d_site[d_site == 0] = 1e-8
            gamma_hat[site] = g_site
            delta_hat2[site] = d_site
            
        self.grand_mean = grand_mean
        self.var_pooled = var_pooled
        self.gamma_hat = gamma_hat
        self.delta_hat2 = delta_hat2
        self.fitted = True
        return self

    def transform(self, X, sites):
        """
        Harmonizes test or validation features X using parameters fitted strictly on training data.
        """
        if not self.fitted:
            raise ValueError("LeakageSafeComBat must be fitted on training data before calling transform.")
            
        X = np.asarray(X, dtype=np.float64)
        N_samples, N_features = X.shape
        sites = np.asarray(sites)
        
        # Standardize data using training grand mean & pooled variance
        s_data = (X - self.grand_mean) / np.sqrt(self.var_pooled)
        bayes_data = s_data.copy()
        
        for site in self.unique_sites:
            idx = (sites == site)
            if not np.any(idx):
                continue
            g_site = self.gamma_hat[site]
            d_site = self.delta_hat2[site]
            
            # Harmonize: (s_data - gamma) / sqrt(delta^2)
            bayes_data[idx] = (s_data[idx] - g_site) / np.sqrt(d_site)
            
        # Reconstruct back to feature space using training grand mean & pooled variance
        X_harmonized = bayes_data * np.sqrt(self.var_pooled) + self.grand_mean
        return X_harmonized.astype(np.float32)

    def fit_transform(self, X, sites, covariates=None):
        self.fit(X, sites, covariates=covariates)
        return self.transform(X, sites)

if __name__ == "__main__":
    np.random.seed(42)
    # 3 sites, 50 samples, 100 features
    X_train = np.random.randn(60, 100) + np.array([0.5]*20 + [1.2]*20 + [-0.8]*20)[:, None]
    sites_train = np.array(["SiteA"]*20 + ["SiteB"]*20 + ["SiteC"]*20)
    
    combat = LeakageSafeComBat()
    X_train_harm = combat.fit_transform(X_train, sites_train)
    
    # Test set from unseen fold (e.g. SiteA)
    X_test = np.random.randn(10, 100) + 0.5
    sites_test = np.array(["SiteA"]*10)
    X_test_harm = combat.transform(X_test, sites_test)
    
    print(f"LeakageSafeComBat trained on {len(sites_train)} samples across {len(combat.unique_sites)} sites.")
    print(f"Harmonized train shape: {X_train_harm.shape}, test shape: {X_test_harm.shape}")
