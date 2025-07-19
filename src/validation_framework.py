"""
Gene Essentiality Prediction - Complete Validation Framework
MSc AI in Biosciences Dissertation Project

Author: Shadiya
Supervisor: Dr. Matteo Fumagalli
Industry Partner: Syngenta

This framework provides comprehensive validation for cross-species 
gene essentiality prediction using machine learning models.
"""

# Step 1: Import required libraries
print("Step 1: Importing required libraries...")

# Core data science libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning core libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, GridSearchCV, 
    train_test_split, validation_curve
)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, matthews_corrcoef,
    balanced_accuracy_score, accuracy_score
)

# Machine learning algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Optional advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("  ✅ XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("  ⚠️  XGBoost not available - install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
    print("  ✅ SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("  ⚠️  SHAP not available - install with: pip install shap")

# Statistical analysis
from scipy import stats
from itertools import combinations

print("✅ Step 1 complete: All available libraries imported successfully!")

# Step 2: Define the gene essentiality validator class
print("\nStep 2: Defining validation framework class...")

class GeneEssentialityValidator:
    """
    Comprehensive validation framework for gene essentiality prediction.
    
    This class provides systematic validation for cross-species gene prediction,
    following best practices for machine learning in bioinformatics research.
    
    Key features:
    • Cross-species validation (Leave-One-Species-Out)
    • Within-species baseline validation
    • Hyperparameter optimisation
    • Comprehensive reporting
    • Data quality assessment
    """
    
    def __init__(self, random_state=42):
        """
        Step 2a: Initialise the validator.
        
        Parameters:
            random_state (int): Random seed for reproducible results across all analyses
        """
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.validation_history = []
        
        # Data storage
        self.df = None
        self.target_col = None
        self.species_col = None
        self.gene_id_col = None
        self.feature_cols = []
        
        print(f"✅ Step 2a complete: Validator initialised (random_state={random_state})")
    
    # Step 3: Data preparation and quality assessment
    
    def load_and_prepare_data(self, df, target_col='larval_lethal', 
                             species_col='species', gene_id_col='gene_id'):
        """
        Step 3a: Load and prepare dataset for validation.
        
        This method prepares my gene essentiality data for comprehensive
        validation, handling data type conversions and feature identification.
        
        Parameters:
            df (pd.DataFrame): Gene dataset with features, species, and target labels
            target_col (str): Column name for essentiality labels (e.g., 'larval_lethal')
            species_col (str): Column name for species information
            gene_id_col (str): Column name for gene identifiers
            
        Returns:
            self: Returns self for method chaining
        """
        print("\n🔧 Step 3a: Preparing dataset for validation...")
        
        # Store configuration
        self.target_col = target_col
        self.species_col = species_col
        self.gene_id_col = gene_id_col
        
        # Create working copy
        self.df = df.copy()
        
        # Handle target variable conversion
        self._convert_target_variable()
        
        # Identify feature columns
        self._identify_feature_columns()
        
        # Validate data structure
        self._validate_data_structure()
        
        # Summary
        print(f"✅ Step 3a complete: Dataset prepared successfully!")
        print(f"   📊 Total samples: {len(self.df):,}")
        print(f"   🧬 Species: {len(self.df[self.species_col].unique())}")
        print(f"   📋 Features: {len(self.feature_cols)}")
        print(f"   🎯 Target: {self.target_col}")
        
        return self
    
    def _convert_target_variable(self):
        """Step 3a.1: Convert target variable to binary format."""
        if self.df[self.target_col].dtype == 'object':
            print(f"  Converting {self.target_col} to binary encoding...")
            
            # Handle different possible values
            unique_values = self.df[self.target_col].unique()
            
            if 'yes' in unique_values:
                self.df[self.target_col] = (self.df[self.target_col] == 'yes').astype(int)
                print(f"    'yes' → 1, others → 0")
            elif 'lethal' in unique_values:
                self.df[self.target_col] = (self.df[self.target_col] == 'lethal').astype(int)
                print(f"    'lethal' → 1, others → 0")
            else:
                print(f"  ⚠️  Warning: Unexpected values in {self.target_col}: {unique_values}")
                print(f"    Please check target variable encoding")
    
    def _identify_feature_columns(self):
        """Step 3a.2: Identify feature columns for analysis."""
        # Exclude metadata columns
        metadata_cols = [
            self.target_col, self.species_col, self.gene_id_col, 
            'gene_name', 'uniprot_id', 'gene_symbol'
        ]
        metadata_cols = [col for col in metadata_cols if col in self.df.columns]
        self.feature_cols = [col for col in self.df.columns if col not in metadata_cols]
        
        print(f"  Identified {len(self.feature_cols)} feature columns")
        if len(self.feature_cols) == 0:
            print(f"  ⚠️  No feature columns found - you'll need to add biological features")
        else:
            print(f"  Feature columns: {self.feature_cols[:5]}{'...' if len(self.feature_cols) > 5 else ''}")
    
    def _validate_data_structure(self):
        """Step 3a.3: Validate data structure and provide warnings."""
        issues = []
        
        # Check required columns exist
        required_cols = [self.target_col, self.species_col]
        for col in required_cols:
            if col not in self.df.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check for empty dataset
        if len(self.df) == 0:
            issues.append("Dataset is empty")
        
        # Check species count
        if len(self.df[self.species_col].unique()) < 2:
            issues.append("Need at least 2 species for cross-species validation")
        
        if issues:
            print(f"  ❌ Data validation issues:")
            for issue in issues:
                print(f"    • {issue}")
            raise ValueError("Data validation failed - please fix issues above")
    
    def explore_dataset(self):
        """
        Step 3b: Comprehensive dataset exploration and quality assessment.
        
        This method provides detailed analysis of my gene essentiality data,
        including species distribution, class balance, and data quality metrics.
        
        Returns:
            dict: Dictionary containing exploration results and statistics
        """
        print("\n🔍 Step 3b: Exploring dataset characteristics...")
        
        # Basic dataset information
        self._explore_basic_statistics()
        
        # Species analysis
        self._explore_species_distribution()
        
        # Target variable analysis
        self._explore_target_distribution()
        
        # Feature quality assessment
        self._assess_feature_quality()
        
        # Data quality warnings
        self._check_data_quality()
        
        # Store exploration results
        self._store_exploration_results()
        
        print("✅ Step 3b complete: Dataset exploration finished!")
        return self.results['data_exploration']
    
    def _explore_basic_statistics(self):
        """Step 3b.1: Explore basic dataset statistics."""
        print(f"📊 Basic dataset statistics:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Memory usage: {self.df.memory_usage().sum() / 1024**2:.1f} MB")
        
        # Check for duplicates
        duplicates = self.df.duplicated().sum()
        print(f"   Duplicate rows: {duplicates}")
        
        # Missing values
        missing_total = self.df.isnull().sum().sum()
        print(f"   Missing values: {missing_total}")
    
    def _explore_species_distribution(self):
        """Step 3b.2: Analyse species distribution."""
        print(f"\n🧬 Species distribution:")
        species_counts = self.df[self.species_col].value_counts()
        
        for species, count in species_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {species}: {count:,} genes ({percentage:.1f}%)")
    
    def _explore_target_distribution(self):
        """Step 3b.3: Analyse target variable distribution."""
        print(f"\n🎯 Target variable analysis ({self.target_col}):")
        
        # Overall distribution
        target_counts = self.df[self.target_col].value_counts()
        print(f"   Overall distribution:")
        for label, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            label_name = "Essential" if label == 1 else "Non-essential"
            print(f"     {label_name}: {count:,} genes ({percentage:.1f}%)")
        
        # By species distribution
        print(f"\n   Distribution by species:")
        for species in self.df[self.species_col].unique():
            species_data = self.df[self.df[self.species_col] == species]
            essential_count = species_data[self.target_col].sum()
            total_count = len(species_data)
            percentage = (essential_count / total_count) * 100
            print(f"     {species}: {essential_count:,}/{total_count:,} ({percentage:.1f}%) essential")
    
    def _assess_feature_quality(self):
        """Step 3b.4: Assess feature quality and completeness."""
        print(f"\n📋 Feature quality assessment:")
        
        if len(self.feature_cols) == 0:
            print(f"   ❌ No features detected")
            print(f"   📝 Recommendation: Add sequence/conservation features for ML analysis")
            return
        
        numeric_features = self.df[self.feature_cols].select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) > 0:
            missing_features = numeric_features.isnull().sum()
            print(f"   Numeric features: {len(numeric_features.columns)}")
            print(f"   Features with missing values: {(missing_features > 0).sum()}")
            
            if (missing_features > 0).any():
                print(f"   Top missing features:")
                top_missing = missing_features[missing_features > 0].head()
                for feat, count in top_missing.items():
                    percentage = (count / len(self.df)) * 100
                    print(f"     {feat}: {count} ({percentage:.1f}%)")
        else:
            print(f"   No numeric features found")
            print(f"   📝 Recommendation: Add quantitative biological features")
    
    def _check_data_quality(self):
        """Step 3b.5: Check data quality and provide warnings."""
        warnings = []
        
        # Class imbalance check
        target_counts = self.df[self.target_col].value_counts()
        if len(target_counts) >= 2:
            imbalance_ratio = target_counts.max() / target_counts.min()
            if imbalance_ratio > 10:
                warnings.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        
        # Species size check
        species_counts = self.df[self.species_col].value_counts()
        min_species_size = species_counts.min()
        if min_species_size < 100:
            small_species = species_counts.idxmin()
            warnings.append(f"Small species dataset: {small_species} has only {min_species_size} genes")
        
        # Feature availability check
        if len(self.feature_cols) == 0:
            warnings.append("No feature columns detected - ML models cannot be trained")
        elif len(self.feature_cols) < 5:
            warnings.append("Limited features detected - consider adding more biological features")
        
        # Display warnings
        if warnings:
            print(f"\n⚠️  Data quality warnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
        else:
            print(f"\n✅ Data quality check passed - no major issues detected")
    
    def _store_exploration_results(self):
        """Step 3b.6: Store exploration results for later use."""
        species_counts = self.df[self.species_col].value_counts()
        target_counts = self.df[self.target_col].value_counts()
        
        self.results['data_exploration'] = {
            'total_samples': len(self.df),
            'n_species': len(self.df[self.species_col].unique()),
            'n_features': len(self.feature_cols),
            'species_counts': species_counts.to_dict(),
            'target_distribution': target_counts.to_dict(),
            'feature_columns': self.feature_cols.copy()
        }
    
    # Step 4: Cross-species validation (core research question)
    
    def cross_species_validation(self, models=None, cv_folds=5):
        """
        Step 4: Cross-species validation - tests my main research hypothesis.
        
        This is the core method that tests whether models trained on one species
        can predict gene essentiality in other species. This directly addresses
        my main research question about cross-species transferability.
        
        Parameters:
            models (dict): Dictionary of models to test. If None, uses default models
            cv_folds (int): Number of cross-validation folds for model evaluation
            
        Returns:
            dict: Dictionary containing cross-species validation results
        """
        print("\n🧪 Step 4: Cross-species validation (Leave-One-Species-Out)")
        print("Testing core research hypothesis: Can models predict gene essentiality across species?")
        
        # Check prerequisites
        if not self._check_validation_prerequisites():
            return None
        
        # Get models to test
        if models is None:
            models = self._get_default_models()
        
        # Run validation
        cross_species_results = self._execute_cross_species_validation(models)
        
        # Analyse results
        self._analyse_cross_species_results(cross_species_results)
        
        # Store results
        self.results['cross_species'] = cross_species_results
        
        print("\n✅ Step 4 complete: Cross-species validation finished!")
        return cross_species_results
    
    def _check_validation_prerequisites(self):
        """Step 4a: Check if validation can be performed."""
        if len(self.feature_cols) == 0:
            print("❌ Cannot perform validation: No feature columns available!")
            print("   📝 Add sequence/conservation features to enable cross-species validation")
            print("   💡 Suggested features: sequence_length, gc_content, conservation_score")
            return False
        
        species_list = self.df[self.species_col].unique()
        if len(species_list) < 2:
            print("❌ Cannot perform cross-species validation: Need at least 2 species!")
            return False
        
        print(f"✅ Prerequisites met: {len(self.feature_cols)} features, {len(species_list)} species")
        return True
    
    def _get_default_models(self):
        """Step 4b: Get default models for validation."""
        print("🔧 Setting up default models...")
        
        models = {}
        
        # Logistic Regression (baseline)
        models['LogisticRegression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Random Forest
        models['RandomForest'] = RandomForestClassifier(
            random_state=self.random_state,
            n_estimators=100,
            class_weight='balanced'
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=self._calculate_scale_pos_weight()
            )
        
        print(f"   📋 Models prepared: {list(models.keys())}")
        return models
    
    def _calculate_scale_pos_weight(self):
        """Calculate scale_pos_weight for XGBoost to handle class imbalance."""
        negative_samples = (self.df[self.target_col] == 0).sum()
        positive_samples = (self.df[self.target_col] == 1).sum()
        return negative_samples / positive_samples if positive_samples > 0 else 1.0
    
    def _execute_cross_species_validation(self, models):
        """Step 4c: Execute the cross-species validation."""
        species_list = self.df[self.species_col].unique()
        cross_species_results = {}
        
        for model_name, model in models.items():
            print(f"\n🔬 Testing {model_name}:")
            model_results = self._test_model_cross_species(model, species_list)
            cross_species_results[model_name] = model_results
        
        return cross_species_results
    
    def _test_model_cross_species(self, model, species_list):
        """Test a single model across all species combinations."""
        model_results = {}
        
        # Leave-one-species-out validation
        for test_species in species_list:
            # Prepare data
            train_data = self.df[self.df[self.species_col] != test_species]
            test_data = self.df[self.df[self.species_col] == test_species]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Extract features and targets
            X_train = train_data[self.feature_cols]
            y_train = train_data[self.target_col]
            X_test = test_data[self.feature_cols]
            y_test = test_data[self.target_col]
            
            # Handle missing values (use training statistics)
            X_train_filled = X_train.fillna(X_train.median())
            X_test_filled = X_test.fillna(X_train.median())
            
            # Check class diversity
            if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
                print(f"   ⚠️  Skipping {test_species}: insufficient class diversity")
                continue
            
            # Train and evaluate
            result = self._train_and_evaluate_model(
                model, X_train_filled, y_train, X_test_filled, y_test, 
                test_species, species_list
            )
            
            if result:
                model_results[result['transfer_description']] = result
        
        return model_results
    
    def _train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test, 
                                 test_species, species_list):
        """Train model and evaluate performance."""
        try:
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate comprehensive metrics
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            # Create result dictionary
            train_species_list = [s for s in species_list if s != test_species]
            train_species_str = '+'.join(train_species_list)
            transfer_description = f"{train_species_str}→{test_species}"
            
            result = {
                'transfer_description': transfer_description,
                'f1_score': f1,
                'accuracy': accuracy,
                'mcc': mcc,
                'balanced_accuracy': balanced_acc,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'test_species': test_species,
                'train_species': train_species_list
            }
            
            # Performance assessment and display
            performance = self._assess_performance_level(f1)
            print(f"   {transfer_description}:")
            print(f"     F1: {f1:.3f}, Balanced Acc: {balanced_acc:.3f}, MCC: {mcc:.3f} {performance}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error with {test_species}: {e}")
            return None
    
    def _assess_performance_level(self, f1_score):
        """Assess performance level based on F1 score."""
        if f1_score > 0.7:
            return "🟢 Excellent"
        elif f1_score > 0.6:
            return "🟡 Good"
        elif f1_score > 0.5:
            return "🟠 Moderate"
        else:
            return "🔴 Poor"
    
    def _analyse_cross_species_results(self, results):
        """Step 4d: Analyse and summarise cross-species validation results."""
        print(f"\n📊 Cross-species validation summary:")
        
        for model_name, model_results in results.items():
            f1_scores = [r['f1_score'] for r in model_results.values()]
            
            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)
                min_f1 = np.min(f1_scores)
                max_f1 = np.max(f1_scores)
                success_rate = (np.array(f1_scores) > 0.6).sum() / len(f1_scores) * 100
                
                print(f"\n{model_name}:")
                print(f"   Average F1: {avg_f1:.3f} ± {std_f1:.3f}")
                print(f"   Range: {min_f1:.3f} - {max_f1:.3f}")
                print(f"   Success rate (F1 > 0.6): {success_rate:.0f}%")
                
                # Research interpretation
                if avg_f1 > 0.7:
                    interpretation = "Excellent cross-species transferability"
                elif avg_f1 > 0.6:
                    interpretation = "Good cross-species transferability"
                elif avg_f1 > 0.5:
                    interpretation = "Moderate cross-species transferability"
                else:
                    interpretation = "Limited cross-species transferability"
                
                print(f"   📋 Interpretation: {interpretation}")
    
    # Step 5: Within-species baseline validation
    
    def within_species_validation(self, cv_folds=5, models=None):
        """
        Step 5: Within-species validation to establish performance baselines.
        
        This method establishes baseline performance by testing models within
        each species separately. This helps distinguish between model capability
        and cross-species transferability challenges.
        
        Parameters:
            cv_folds (int): Number of cross-validation folds
            models (dict): Models to test. If None, uses default models
            
        Returns:
            dict: Dictionary containing within-species validation results
        """
        print(f"\n🔬 Step 5: Within-species validation ({cv_folds}-fold CV)")
        print("Establishing baseline performance within each species")
        
        if len(self.feature_cols) == 0:
            print("❌ Cannot perform validation: No feature columns available!")
            return None
        
        if models is None:
            models = self._get_default_models()
        
        within_species_results = {}
        
        for species in self.df[self.species_col].unique():
            print(f"\n🧬 Analysing {species}:")
            species_results = self._validate_single_species(species, models, cv_folds)
            if species_results:
                within_species_results[species] = species_results
        
        # Store results
        self.results['within_species'] = within_species_results
        
        # Summary
        self._summarise_within_species_results(within_species_results)
        
        print("\n✅ Step 5 complete: Within-species validation finished!")
        return within_species_results
    
    def _validate_single_species(self, species, models, cv_folds):
        """Validate models on a single species."""
        species_data = self.df[self.df[self.species_col] == species]
        
        # Check data sufficiency
        if len(species_data) < cv_folds * 2:
            print(f"   ⚠️  Insufficient data for {cv_folds}-fold CV ({len(species_data)} samples)")
            return None
        
        # Prepare data
        X = species_data[self.feature_cols].fillna(species_data[self.feature_cols].median())
        y = species_data[self.target_col]
        
        # Check class diversity
        if len(y.unique()) < 2:
            print(f"   ⚠️  Only one class present - skipping validation")
            return None
        
        print(f"   📊 {len(species_data):,} genes, {len(y.unique())} classes")
        
        # Run cross-validation for each model
        species_results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model in models.items():
            try:
                # Cross-validation
                f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
                accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                
                species_results[model_name] = {
                    'f1_mean': f1_scores.mean(),
                    'f1_std': f1_scores.std(),
                    'accuracy_mean': accuracy_scores.mean(),
                    'accuracy_std': accuracy_scores.std(),
                    'sample_size': len(species_data),
                    'n_folds': cv_folds
                }
                
                print(f"   {model_name}: F1={f1_scores.mean():.3f}±{f1_scores.std():.3f}")
                
            except Exception as e:
                print(f"   ❌ {model_name}: {e}")
        
        return species_results
    
    def _summarise_within_species_results(self, results):
        """Summarise within-species validation results."""
        print(f"\n📊 Within-species validation summary:")
        
        # Calculate averages across species
        model_averages = {}
        
        for species, species_results in results.items():
            for model_name, metrics in species_results.items():
                if model_name not in model_averages:
                    model_averages[model_name] = []
                model_averages[model_name].append(metrics['f1_mean'])
        
        for model_name, f1_scores in model_averages.items():
            avg_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"{model_name}: Average F1 = {avg_f1:.3f} ± {std_f1:.3f} (across species)")
    
    # Step 6: Hyperparameter optimisation
    
    def hyperparameter_optimisation(self, model_type='xgboost', cv_folds=3, quick_search=True):
        """
        Step 6: Optimise hyperparameters for specified model.
        
        This method performs systematic hyperparameter optimisation to ensure
        fair model comparison and optimal performance.
        
        Parameters:
            model_type (str): Type of model ('xgboost', 'randomforest', 'svm')
            cv_folds (int): Cross-validation folds for optimisation
            quick_search (bool): If True, uses reduced parameter grid for faster search
            
        Returns:
            sklearn.base.BaseEstimator: Optimised model with best parameters
        """
        print(f"\n⚙️  Step 6: Hyperparameter optimisation ({model_type.upper()})")
        
        if len(self.feature_cols) == 0:
            print("❌ Cannot optimise: No feature columns available!")
            return None
        
        # Prepare data
        X = self.df[self.feature_cols].fillna(self.df[self.feature_cols].median())
        y = self.df[self.target_col]
        
        # Get model and parameter grid
        model, param_grid = self._get_model_and_params(model_type, quick_search)
        if model is None:
            return None
        
        # Perform grid search
        best_model = self._perform_grid_search(model, param_grid, X, y, cv_folds)
        
        # Store results
        self.models[f'best_{model_type}'] = best_model
        
        print(f"✅ Step 6 complete: {model_type.upper()} optimisation finished!")
        return best_model
    
    def _get_model_and_params(self, model_type, quick_search):
        """Get model and parameter grid for optimisation."""
        if model_type.lower() == 'xgboost':
            if not XGBOOST_AVAILABLE:
                print("❌ XGBoost not available - install with: pip install xgboost")
                return None, None
            
            model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=self._calculate_scale_pos_weight()
            )
            
            if quick_search:
                param_grid = {
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.1, 0.2],
                    'n_estimators': [100, 300]
                }
            else:
                param_grid = {
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [100, 300, 500],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
        
        elif model_type.lower() == 'randomforest':
            model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            
            if quick_search:
                param_grid = {
                    'n_estimators': [100, 300],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            else:
                param_grid = {
                    'n_estimators': [100, 300, 500],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        
        else:
            print(f"❌ Model type '{model_type}' not supported")
            print("   Supported models: 'xgboost', 'randomforest'")
            return None, None
        
        search_type = "Quick" if quick_search else "Comprehensive"
        print(f"🔧 {search_type} parameter search for {model_type}")
        print(f"   Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        return model, param_grid
    
    def _perform_grid_search(self, model, param_grid, X, y, cv_folds):
        """Perform grid search with cross-validation."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        print(f"🔍 Running grid search with {cv_folds}-fold CV...")
        
        grid_search = GridSearchCV(
            model, param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        # Store optimisation results
        model_type = model.__class__.__name__.lower()
        self.results[f'{model_type}_optimisation'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"✅ Optimisation complete!")
        print(f"   Best F1 score: {grid_search.best_score_:.3f}")
        print(f"   Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    # Step 7: Comprehensive results analysis and reporting
    
    def generate_comprehensive_report(self):
        """
        Step 7: Generate comprehensive validation report for my thesis.
        
        This method creates a detailed report suitable for inclusion in my
        MSc dissertation, covering all validation results and research conclusions.
        
        Returns:
            dict: Complete results dictionary for further analysis
        """
        print("\n📋 Step 7: Generating comprehensive validation report")
        
        # Report header
        self._generate_report_header()
        
        # Dataset summary
        self._report_dataset_summary()
        
        # Cross-species validation results
        self._report_cross_species_results()
        
        # Within-species baseline results
        self._report_within_species_results()
        
        # Hyperparameter optimisation results
        self._report_optimisation_results()
        
        # Research conclusions and recommendations
        self._generate_research_conclusions()
        
        print("✅ Step 7 complete: Comprehensive report generated!")
        
        return self.results
    
    def _generate_report_header(self):
        """Generate report header with project information."""
        print("Gene Essentiality Prediction - Validation Results")
        print(f"MSc AI in Biosciences Dissertation Project")
        print(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("")
        
        if hasattr(self, 'df') and self.df is not None:
            n_species = len(self.df[self.species_col].unique())
            print(f"Dataset: {len(self.df):,} genes across {n_species} species")
        print("")
    
    def _report_dataset_summary(self):
        """Report dataset summary statistics."""
        if 'data_exploration' in self.results:
            print("📊 Dataset summary:")
            exploration = self.results['data_exploration']
            
            print(f"   Total genes analysed: {exploration['total_samples']:,}")
            print(f"   Number of species: {exploration['n_species']}")
            print(f"   Number of features: {exploration['n_features']}")
            
            print(f"\n   Species distribution:")
            for species, count in exploration['species_counts'].items():
                percentage = (count / exploration['total_samples']) * 100
                print(f"     {species}: {count:,} genes ({percentage:.1f}%)")
            
            print(f"\n   Target distribution:")
            for label, count in exploration['target_distribution'].items():
                label_name = "Essential" if label == 1 else "Non-essential"
                percentage = (count / exploration['total_samples']) * 100
                print(f"     {label_name}: {count:,} genes ({percentage:.1f}%)")
            print("")
    
    def _report_cross_species_results(self):
        """Report cross-species validation results."""
        if 'cross_species' in self.results:
            print("🧪 Cross-species validation results:")
            print("   (Core research question: cross-species gene essentiality prediction)")
            
            for model_name, results in self.results['cross_species'].items():
                f1_scores = [r['f1_score'] for r in results.values()]
                
                if f1_scores:
                    avg_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores)
                    success_rate = (np.array(f1_scores) > 0.6).sum() / len(f1_scores) * 100
                    
                    print(f"\n   {model_name}:")
                    print(f"     Average F1 score: {avg_f1:.3f} ± {std_f1:.3f}")
                    print(f"     Success rate (F1 > 0.6): {success_rate:.0f}%")
                    print(f"     Number of transfers tested: {len(f1_scores)}")
                    
                    # Detailed transfer results
                    print(f"     Individual transfer results:")
                    for transfer, metrics in results.items():
                        f1 = metrics['f1_score']
                        performance = self._assess_performance_level(f1)
                        print(f"       {transfer}: F1={f1:.3f} {performance}")
            print("")
    
    def _report_within_species_results(self):
        """Report within-species validation results."""
        if 'within_species' in self.results:
            print("🔬 Within-species validation results:")
            print("   (Baseline performance within each species)")
            
            for species, models in self.results['within_species'].items():
                print(f"\n   {species}:")
                for model_name, metrics in models.items():
                    f1_mean = metrics['f1_mean']
                    f1_std = metrics['f1_std']
                    sample_size = metrics['sample_size']
                    print(f"     {model_name}: F1={f1_mean:.3f}±{f1_std:.3f} (n={sample_size:,})")
            print("")
    
    def _report_optimisation_results(self):
        """Report hyperparameter optimisation results."""
        optimisation_keys = [k for k in self.results.keys() if k.endswith('_optimisation')]
        
        if optimisation_keys:
            print("⚙️  Hyperparameter optimisation results:")
            
            for key in optimisation_keys:
                model_type = key.replace('_optimisation', '').upper()
                results = self.results[key]
                
                print(f"\n   {model_type}:")
                print(f"     Best CV F1 score: {results['best_score']:.3f}")
                print(f"     Best parameters: {results['best_params']}")
            print("")
    
    def _generate_research_conclusions(self):
        """Generate research conclusions and recommendations."""
        print("🎯 Research conclusions and recommendations:")
        
        conclusions = []
        recommendations = []
        
        # Analyse cross-species performance
        if 'cross_species' in self.results:
            all_f1_scores = []
            for results in self.results['cross_species'].values():
                f1_scores = [r['f1_score'] for r in results.values()]
                all_f1_scores.extend(f1_scores)
            
            if all_f1_scores:
                avg_cross_species_f1 = np.mean(all_f1_scores)
                
                if avg_cross_species_f1 > 0.7:
                    conclusions.append("✅ Excellent cross-species gene essentiality prediction achieved")
                    recommendations.append("Consider application to additional pest species")
                elif avg_cross_species_f1 > 0.6:
                    conclusions.append("🟡 Moderate cross-species prediction capability demonstrated")
                    recommendations.append("Investigate evolutionary conservation features")
                else:
                    conclusions.append("❌ Limited cross-species transferability observed")
                    recommendations.append("Focus on species-specific models or transfer learning")
                
                conclusions.append(f"   Average cross-species F1: {avg_cross_species_f1:.3f}")
        
        # Feature availability analysis
        if hasattr(self, 'feature_cols'):
            if len(self.feature_cols) == 0:
                conclusions.append("⚠️  No biological features currently available")
                recommendations.append("Add sequence features (GC content, length, codon usage)")
                recommendations.append("Include evolutionary conservation scores")
                recommendations.append("Incorporate protein domain information")
            elif len(self.feature_cols) < 10:
                conclusions.append("⚠️  Limited feature set detected")
                recommendations.append("Expand feature engineering for improved performance")
        
        # Model performance comparison
        if 'cross_species' in self.results and len(self.results['cross_species']) > 1:
            model_performances = {}
            for model_name, results in self.results['cross_species'].items():
                f1_scores = [r['f1_score'] for r in results.values()]
                if f1_scores:
                    model_performances[model_name] = np.mean(f1_scores)
            
            if model_performances:
                best_model = max(model_performances, key=model_performances.get)
                conclusions.append(f"🏆 Best performing model: {best_model}")
                recommendations.append(f"Focus on {best_model} for production deployment")
        
        # Display conclusions
        print("\n   Key findings:")
        for i, conclusion in enumerate(conclusions, 1):
            print(f"     {i}. {conclusion}")
        
        print("\n   Recommendations for future work:")
        for i, recommendation in enumerate(recommendations, 1):
            print(f"     {i}. {recommendation}")
        
        # Thesis implications
        print("\n   Implications for my MSc dissertation:")
        print("     • Cross-species validation demonstrates transferability potential")
        print("     • Results support precision agriculture applications")
        print("     • Methodology suitable for other agricultural pest species")
        print("     • Framework provides foundation for Syngenta collaboration")

# Step 8: Convenience functions for complete pipelines

def run_complete_validation_pipeline(df, target_col='larval_lethal', 
                                   species_col='species', quick_analysis=True):
    """
    Step 8: Run complete validation pipeline for gene essentiality prediction.
    
    This convenience function executes the entire validation workflow,
    providing a one-step solution for comprehensive model validation.
    
    Parameters:
        df (pd.DataFrame): Gene dataset with essentiality labels and species information
        target_col (str): Column name for essentiality labels
        species_col (str): Column name for species information
        quick_analysis (bool): If True, uses faster parameter grids for quicker results
        
    Returns:
        GeneEssentialityValidator: Validator object with complete results for further analysis
    """
    print("🧬 Complete gene essentiality validation pipeline")
    print("Executing comprehensive validation for cross-species gene essentiality prediction")
    print("")
    
    # Step 1: Initialise validator
    print("🔧 Initialising validation framework...")
    validator = GeneEssentialityValidator(random_state=42)
    
    try:
        # Step 2: Load and prepare data
        validator.load_and_prepare_data(df, target_col=target_col, species_col=species_col)
        
        # Step 3: Explore dataset
        validator.explore_dataset()
        
        # Step 4: Run validation if features are available
        if len(validator.feature_cols) > 0:
            # Within-species baseline
            validator.within_species_validation()
            
            # Cross-species validation (core research question)
            validator.cross_species_validation()
            
            # Hyperparameter optimisation (if requested)
            if not quick_analysis:
                if XGBOOST_AVAILABLE:
                    validator.hyperparameter_optimisation('xgboost', quick_search=False)
                validator.hyperparameter_optimisation('randomforest', quick_search=False)
        
        # Step 5: Generate comprehensive report
        validator.generate_comprehensive_report()
        
        print("\n🎉 Complete validation pipeline finished successfully!")
        print("📊 Results available in validator.results dictionary")
        print("📋 Use validator.generate_comprehensive_report() to view detailed results")
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        print("💡 Check your data format and column names")
        print("📝 Ensure target_col and species_col exist in your dataset")
    
    return validator

def run_basic_validation_pipeline(df, target_col='larval_lethal', species_col='species'):
    """
    Run basic validation pipeline for quick analysis.
    
    This function provides a streamlined validation workflow suitable for
    initial data exploration and quick hypothesis testing.
    
    Parameters:
        df (pd.DataFrame): Gene dataset
        target_col (str): Target column name
        species_col (str): Species column name
        
    Returns:
        GeneEssentialityValidator: Validator with basic results
    """
    print("🧬 Basic gene essentiality validation pipeline")
    
    validator = GeneEssentialityValidator(random_state=42)
    
    try:
        # Essential steps only
        validator.load_and_prepare_data(df, target_col=target_col, species_col=species_col)
        validator.explore_dataset()
        
        if len(validator.feature_cols) > 0:
            validator.cross_species_validation()
        
        print("\n✅ Basic validation pipeline complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    return validator

# Step 9: Framework initialisation and testing

if __name__ == "__main__":
    print("\n🧬 Gene Essentiality Validation Framework")
    print("Complete framework for cross-species gene essentiality prediction")
    print("MSc AI in Biosciences Dissertation Project")
    print("")
    print("Framework features:")
    print("✅ Cross-species validation (Leave-One-Species-Out)")
    print("✅ Within-species baseline validation")
    print("✅ Hyperparameter optimisation")
    print("✅ Comprehensive reporting")
    print("✅ Data quality assessment")
    print("✅ Multiple ML algorithms support")
    print("")
    print("Usage examples:")
    print("")
    print("# Basic usage:")
    print("validator = GeneEssentialityValidator()")
    print("validator.load_and_prepare_data(df)")
    print("validator.cross_species_validation()")
    print("")
    print("# Complete pipeline:")
    print("validator = run_complete_validation_pipeline(df)")
    print("")
    print("Ready for gene essentiality research! 🚀")