"""
Debug logger for FIMEC operations.
Captures coupling matrices, posteriors, and other debug information.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class FIMECDebugLogger:
    """Logger for debugging FIMEC encode/decode operations."""

    def __init__(self, log_dir: str = "./logs", operation: str = "encode"):
        """
        Initialize the debug logger.

        Args:
            log_dir: Directory to save logs
            operation: "encode" or "decode"
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.operation = operation
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Storage
        self.logs = {
            'metadata': {
                'timestamp': self.timestamp,
                'operation': operation,
            },
            'couplings': [],
            'tokens': [],
            'summary': {}
        }

        self.coupling_count = 0
        self.token_count = 0
        self._original_greedy_mec = None
        self._is_patched = False

    def start(self, **kwargs):
        """Start logging with metadata."""
        self.logs['metadata'].update(kwargs)
        self._setup_patch()
        print(f"\n{'='*70}")
        print(f"DEBUG LOGGING ENABLED: {self.operation.upper()}")
        print(f"Log directory: {self.log_dir}")
        print(f"{'='*70}\n")

    def _setup_patch(self):
        """Monkey-patch greedy_mec to capture coupling matrices."""
        if self._is_patched:
            return

        from mec.approximation import algorithms

        # Save original
        self._original_greedy_mec = algorithms.greedy_mec

        def logged_greedy_mec(p, q):
            # Call original
            result = self._original_greedy_mec(p, q)

            # Log it
            self._log_coupling(result, p, q)

            return result

        # Replace
        algorithms.greedy_mec = logged_greedy_mec
        self._is_patched = True

    def _log_coupling(self, coupling: np.ndarray, token_dist: np.ndarray, byte_dist: np.ndarray):
        """Log coupling matrix information."""

        # Calculate entropies
        token_entropy = float(-np.sum(token_dist * np.log2(token_dist + 1e-10)))
        byte_entropy = float(-np.sum(byte_dist * np.log2(byte_dist + 1e-10)))

        coupling_info = {
            'coupling_num': self.coupling_count,
            'iteration': self.token_count,
            'shape': list(coupling.shape),
            'token_dist': {
                'entropy': round(token_entropy, 4),
                'max_prob': round(float(token_dist.max()), 4),
                'min_prob': round(float(token_dist.min()), 4),
                'peaked': token_dist.max() > 0.7,
                'top_5_probs': [round(float(p), 4) for p in sorted(token_dist, reverse=True)[:5]],
            },
            'byte_dist': {
                'entropy': round(byte_entropy, 4),
                'max_prob': round(float(byte_dist.max()), 4),
                'argmax': int(byte_dist.argmax()),
                'top_5_indices': [int(i) for i in np.argsort(byte_dist)[::-1][:5]],
                'top_5_probs': [round(float(byte_dist[i]), 4) for i in np.argsort(byte_dist)[::-1][:5]],
            },
            'coupling_stats': {
                'sparsity': round(float(np.mean(coupling == 0)), 4),
                'max': round(float(coupling.max()), 6),
                'mean': round(float(coupling.mean()), 6),
                'std': round(float(coupling.std()), 6),
            }
        }

        self.logs['couplings'].append(coupling_info)

        # Print progress
        print(f"  Coupling {self.coupling_count:3d}: "
              f"Token H={token_entropy:.2f} (max_p={token_dist.max():.3f}), "
              f"Byte H={byte_entropy:.2f} (argmax={byte_dist.argmax():3d})")

        self.coupling_count += 1
        self.token_count += 1

    def log_result(self, result_data: Dict[str, Any]):
        """Log final results."""
        self.logs.update(result_data)

    def finish(self):
        """Finalize and save logs."""

        # Generate summary
        if self.logs['couplings']:
            token_entropies = [c['token_dist']['entropy'] for c in self.logs['couplings']]
            byte_entropies = [c['byte_dist']['entropy'] for c in self.logs['couplings']]

            self.logs['summary'] = {
                'total_couplings': len(self.logs['couplings']),
                'total_tokens': self.token_count,
                'token_entropy': {
                    'mean': round(float(np.mean(token_entropies)), 4),
                    'min': round(float(np.min(token_entropies)), 4),
                    'max': round(float(np.max(token_entropies)), 4),
                    'std': round(float(np.std(token_entropies)), 4),
                },
                'byte_entropy': {
                    'mean': round(float(np.mean(byte_entropies)), 4),
                    'min': round(float(np.min(byte_entropies)), 4),
                    'max': round(float(np.max(byte_entropies)), 4),
                },
                'peaked_tokens_count': sum(1 for c in self.logs['couplings']
                                          if c['token_dist']['peaked']),
                'avg_info_per_token': round(float(np.mean(token_entropies)), 4),
            }

        # Save to JSON
        log_file = self.log_dir / f"{self.operation}_{self.timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

        # Print summary
        print(f"\n{'='*70}")
        print(f"DEBUG SUMMARY")
        print(f"{'='*70}")
        if self.logs['summary']:
            for key, val in self.logs['summary'].items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k2, v2 in val.items():
                        print(f"    {k2:15s}: {v2}")
                else:
                    print(f"  {key:25s}: {val}")

        print(f"\n✓ Saved debug logs to: {log_file}")
        print(f"{'='*70}\n")

        # Restore original greedy_mec
        if self._is_patched and self._original_greedy_mec:
            from mec.approximation import algorithms
            algorithms.greedy_mec = self._original_greedy_mec
            self._is_patched = False

        return log_file
