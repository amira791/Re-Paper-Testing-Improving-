"""
PHASE 3: ADVANCED FEATURES & MODEL EVALUATION
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, model_manager, dataloaders, data_config):
        self.model_manager = model_manager
        self.dataloaders = dataloaders
        self.data_config = data_config
        self.device = model_manager.device
        
    def evaluate_per_vehicle(self):
        """Evaluate model performance per vehicle"""
        
        print("\n" + "="*60)
        print("PER-VEHICLE EVALUATION")
        print("="*60)
        
        vehicle_results = {}
        
        # Get test dataset
        test_dataset = self.dataloaders['test_dataset']
        
        # Group by vehicle_id from metadata
        vehicle_predictions = {}
        vehicle_targets = {}
        
        # We need to process sequences and extract vehicle info
        self.model_manager.model.eval()
        
        with torch.no_grad():
            for batch in self.dataloaders['test_loader']:
                sequences = batch['sequence'].to(self.device)
                targets = batch['target'].cpu().numpy()
                metadata = batch['metadata']
                
                # Get predictions
                predictions, uncertainties = self.model_manager.model(sequences)
                predictions = predictions.cpu().numpy().flatten()
                
                # Group by vehicle
                for i, meta in enumerate(metadata):
                    vehicle_id = meta['vehicle_id']
                    
                    if vehicle_id not in vehicle_predictions:
                        vehicle_predictions[vehicle_id] = []
                        vehicle_targets[vehicle_id] = []
                    
                    vehicle_predictions[vehicle_id].append(predictions[i])
                    vehicle_targets[vehicle_id].append(targets[i])
        
        # Calculate metrics per vehicle
        results = []
        for vehicle_id in vehicle_predictions.keys():
            preds = np.array(vehicle_predictions[vehicle_id])
            tgts = np.array(vehicle_targets[vehicle_id])
            
            mae = mean_absolute_error(tgts, preds)
            rmse = np.sqrt(mean_squared_error(tgts, preds))
            r2 = r2_score(tgts, preds)
            
            results.append({
                'vehicle_id': vehicle_id,
                'samples': len(preds),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mean_target': np.mean(tgts),
                'std_target': np.std(tgts)
            })
            
            print(f"Vehicle {vehicle_id}:")
            print(f"  Samples: {len(preds):,}")
            print(f"  MAE: {mae:.3f}%")
            print(f"  RMSE: {rmse:.3f}%")
            print(f"  R²: {r2:.3f}")
            print(f"  Target mean: {np.mean(tgts):.1f}% ± {np.std(tgts):.1f}%")
            print()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def visualize_predictions(self, test_results, num_samples=100):
        """Visualize predictions vs targets"""
        
        predictions = test_results['predictions']
        targets = test_results['targets']
        uncertainties = test_results['uncertainties']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Predictions vs Targets scatter
        ax1 = axes[0, 0]
        ax1.scatter(targets[:num_samples], predictions[:num_samples], 
                   alpha=0.6, s=20, c='steelblue')
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        ax1.set_xlabel('Actual SOH (%)')
        ax1.set_ylabel('Predicted SOH (%)')
        ax1.set_title('Predictions vs Actual Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² text
        r2 = r2_score(targets, predictions)
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=ax1.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residual plot
        ax2 = axes[0, 1]
        residuals = predictions - targets
        ax2.scatter(predictions[:num_samples], residuals[:num_samples], 
                   alpha=0.6, s=20, c='coral')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        
        ax2.set_xlabel('Predicted SOH (%)')
        ax2.set_ylabel('Residual (Predicted - Actual)')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution
        ax3 = axes[0, 2]
        errors = np.abs(residuals)
        ax3.hist(errors, bins=50, alpha=0.7, color='seagreen', edgecolor='black')
        ax3.axvline(x=np.mean(errors), color='r', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
        ax3.axvline(x=np.median(errors), color='orange', linestyle='--',
                   linewidth=2, label=f'Median: {np.median(errors):.3f}')
        
        ax3.set_xlabel('Absolute Error (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Uncertainty visualization
        ax4 = axes[1, 0]
        # Sort by uncertainty for visualization
        sorted_indices = np.argsort(uncertainties)[:num_samples]
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_predictions = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        x_positions = np.arange(len(sorted_uncertainties))
        ax4.errorbar(x_positions, sorted_predictions, 
                    yerr=1.96*sorted_uncertainties, 
                    fmt='o', alpha=0.6, capsize=3, 
                    label='Prediction ± 95% CI')
        ax4.scatter(x_positions, sorted_targets, color='red', 
                   s=30, alpha=0.7, label='Actual', zorder=5)
        
        ax4.set_xlabel('Sample Index (sorted by uncertainty)')
        ax4.set_ylabel('SOH (%)')
        ax4.set_title('Predictions with Uncertainty Intervals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Time series prediction example
        ax5 = axes[1, 1]
        # Select a continuous sequence from test set
        test_dataset = self.dataloaders['test_dataset']
        
        # Find sequences from same vehicle/session
        session_sequences = []
        session_targets = []
        
        # Group by vehicle and session from metadata
        sequence_data = {}
        for i in range(min(100, len(test_dataset))):
            metadata = test_dataset.metadata[i]
            key = f"{metadata['vehicle_id']}_{metadata['session_id']}"
            
            if key not in sequence_data:
                sequence_data[key] = {'predictions': [], 'targets': [], 'indices': []}
            
            sequence_data[key]['predictions'].append(test_dataset.targets[i])
            sequence_data[key]['targets'].append(test_dataset.targets[i])
            sequence_data[key]['indices'].append(metadata['start_idx'])
        
        # Plot the sequence with most points
        best_key = max(sequence_data.keys(), 
                      key=lambda k: len(sequence_data[k]['predictions']))
        
        data = sequence_data[best_key]
        sorted_indices = np.argsort(data['indices'])
        
        predictions_sorted = np.array(data['predictions'])[sorted_indices]
        targets_sorted = np.array(data['targets'])[sorted_indices]
        
        ax5.plot(targets_sorted, 'b-', alpha=0.7, linewidth=2, label='Actual SOH')
        ax5.plot(predictions_sorted, 'r--', alpha=0.7, linewidth=2, label='Predicted SOH')
        
        ax5.set_xlabel('Time Step in Session')
        ax5.set_ylabel('SOH (%)')
        ax5.set_title(f'SOH Prediction in Session {best_key}')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature importance (through gradient analysis)
        ax6 = axes[1, 2]
        
        # Get a sample batch for gradient analysis
        sample_batch = next(iter(self.dataloaders['train_loader']))
        sample_seq = sample_batch['sequence'][:1].to(self.device)
        sample_seq.requires_grad = True
        
        self.model_manager.model.eval()
        prediction, _ = self.model_manager.model(sample_seq)
        
        # Calculate gradients
        prediction.backward()
        gradients = sample_seq.grad.abs().mean(dim=[0, 1]).cpu().numpy()
        
        # Get feature names
        feature_names = self.data_config['feature_names']
        
        # Sort by importance
        indices = np.argsort(gradients)[::-1][:10]  # Top 10
        sorted_features = [feature_names[i] for i in indices]
        sorted_gradients = gradients[indices]
        
        ax6.barh(range(len(sorted_features)), sorted_gradients, 
                color='purple', alpha=0.7)
        ax6.set_yticks(range(len(sorted_features)))
        ax6.set_yticklabels(sorted_features)
        ax6.set_xlabel('Average Gradient Magnitude')
        ax6.set_title('Feature Importance (Gradient-based)')
        ax6.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Mean Absolute Error (MAE): {np.mean(np.abs(residuals)):.3f}%")
        print(f"Root Mean Square Error (RMSE): {np.sqrt(np.mean(residuals**2)):.3f}%")
        print(f"R² Score: {r2:.3f}")
        print(f"Mean Uncertainty: {np.mean(uncertainties):.3f}")
        
        # Calculate calibration metrics
        confidence_levels = [0.5, 0.8, 0.9, 0.95]
        for conf in confidence_levels:
            z_score = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96}[conf]
            coverage = np.mean(
                (targets >= predictions - z_score * uncertainties) &
                (targets <= predictions + z_score * uncertainties)
            )
            print(f"Coverage at {conf*100:.0f}% confidence: {coverage:.3f}")
    
    def analyze_failure_cases(self, test_results, threshold=5.0):
        """Analyze cases where model failed (error > threshold %)"""
        
        predictions = test_results['predictions']
        targets = test_results['targets']
        errors = np.abs(predictions - targets)
        
        # Identify failure cases
        failure_mask = errors > threshold
        failure_indices = np.where(failure_mask)[0]
        
        if len(failure_indices) == 0:
            print(f"\nNo failure cases found (threshold: {threshold}%)")
            return None
        
        print(f"\n" + "="*60)
        print(f"FAILURE ANALYSIS (Error > {threshold}%)")
        print("="*60)
        print(f"Total failures: {len(failure_indices)} ({len(failure_indices)/len(predictions)*100:.1f}%)")
        
        # Analyze characteristics of failures
        test_dataset = self.dataloaders['test_dataset']
        
        failure_stats = {
            'max_error': errors.max(),
            'mean_error_failures': errors[failure_mask].mean(),
            'median_error_failures': np.median(errors[failure_mask]),
            'failure_vehicle_distribution': {}
        }
        
        # Check vehicle distribution of failures
        for idx in failure_indices[:10]:  # First 10 failures
            metadata = test_dataset.metadata[idx]
            vehicle_id = metadata['vehicle_id']
            
            if vehicle_id not in failure_stats['failure_vehicle_distribution']:
                failure_stats['failure_vehicle_distribution'][vehicle_id] = 0
            failure_stats['failure_vehicle_distribution'][vehicle_id] += 1
        
        print("\nFailure Statistics:")
        print(f"  Maximum error: {failure_stats['max_error']:.2f}%")
        print(f"  Mean error (failures): {failure_stats['mean_error_failures']:.2f}%")
        print(f"  Median error (failures): {failure_stats['median_error_failures']:.2f}%")
        
        print("\nVehicle distribution of failures:")
        for vehicle_id, count in failure_stats['failure_vehicle_distribution'].items():
            print(f"  Vehicle {vehicle_id}: {count} failures")
        
        return failure_stats
    
    def create_comparison_report(self, baseline_results=None):
        """Create comprehensive comparison report"""
        
        # Get current model results
        test_results = self.model_manager.evaluate(self.dataloaders['test_loader'])
        current_metrics = test_results['metrics']
        
        report = {
            'model_name': 'Hybrid Deep Learning Model',
            'architecture': 'LSTM-CNN-Transformer Hybrid',
            'sequence_length': self.data_config['sequence_length'],
            'num_features': self.data_config['num_features'],
            'performance': current_metrics,
            'training_history': self.model_manager.history,
            'test_set_size': len(self.dataloaders['test_dataset']),
            'feature_names': self.data_config['feature_names']
        }
        
        # Add baseline comparison if provided
        if baseline_results:
            report['baseline_comparison'] = {
                'baseline_mae': baseline_results.get('mae'),
                'improvement_mae': (baseline_results.get('mae') - current_metrics['mae']) / baseline_results.get('mae') * 100,
                'baseline_rmse': baseline_results.get('rmse'),
                'improvement_rmse': (baseline_results.get('rmse') - current_metrics['rmse']) / baseline_results.get('rmse') * 100
            }
        
        # Save report
        import json
        with open('model_comparison_report.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                else:
                    return obj
            
            json.dump(report, f, indent=2, default=convert_for_json)
        
        print(f"\nComparison report saved to model_comparison_report.json")
        return report


def main_evaluation(dataloaders, data_config, model_manager):
    """Main evaluation pipeline"""
    
    print("="*80)
    print("PHASE 3: ADVANCED MODEL EVALUATION")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_manager, dataloaders, data_config)
    
    # 1. Get test results
    test_results = model_manager.evaluate(dataloaders['test_loader'])
    
    # 2. Per-vehicle evaluation
    vehicle_results = evaluator.evaluate_per_vehicle()
    
    # 3. Visualize predictions
    evaluator.visualize_predictions(test_results)
    
    # 4. Analyze failure cases
    failure_stats = evaluator.analyze_failure_cases(test_results, threshold=3.0)
    
    # 5. Create comprehensive report
    report = evaluator.create_comparison_report()
    
    # 6. Save important artifacts
    artifacts = {
        'vehicle_results': vehicle_results,
        'test_results': test_results,
        'failure_stats': failure_stats,
        'report': report
    }
    
    joblib.dump(artifacts, 'evaluation_artifacts.pkl')
    print(f"\nEvaluation artifacts saved to evaluation_artifacts.pkl")
    
    return evaluator, artifacts


if __name__ == "__main__":
    print("This module requires Phase 1 and Phase 2 to be completed first.")
    print("Run main_model_training() from Phase 2 first.")