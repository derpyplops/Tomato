"""
Early patching logger - patches BEFORE any imports.
This ensures coupling matrices are actually captured.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path


class EarlyPatchLogger:
    """
    Logger that MUST be initialized before importing Encoder.
    Patches greedy_mec at module level to ensure it's captured.
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.coupling_count = 0
        self.couplings = []
        self._original_greedy_mec = None
        self._is_patched = False

    def patch_now(self):
        """
        Patch greedy_mec IMMEDIATELY.
        Call this BEFORE importing Encoder.
        """
        if self._is_patched:
            print("⚠️  Already patched")
            return

        # Import the module where greedy_mec is actually USED (not just defined)
        import mec.iterative.algorithms as mec_iter

        # Save original (from the iterative module's namespace)
        self._original_greedy_mec = mec_iter.greedy_mec

        # Create patched version with closure over self
        def logged_greedy_mec(p, q):
            # Call original
            result = self._original_greedy_mec(p, q)

            # Log it
            self._log_coupling(result, p, q)

            return result

        # Replace in the ITERATIVE module (where FIMEC actually calls it)
        mec_iter.greedy_mec = logged_greedy_mec
        self._is_patched = True

        print("✓ Patched greedy_mec in iterative module BEFORE Encoder import")

    def _log_coupling(self, coupling: np.ndarray, token_dist: np.ndarray, byte_dist: np.ndarray):
        """Log coupling matrix information."""

        # Calculate entropies
        token_entropy = float(-np.sum(token_dist * np.log2(token_dist + 1e-10)))
        byte_entropy = float(-np.sum(byte_dist * np.log2(byte_dist + 1e-10)))

        coupling_info = {
            'coupling_num': self.coupling_count,
            'shape': list(coupling.shape),
            'token_dist': {
                'entropy': round(token_entropy, 4),
                'max_prob': round(float(token_dist.max()), 4),
                'min_prob': round(float(token_dist.min()), 4),
                'peaked': bool(token_dist.max() > 0.7),
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

        self.couplings.append(coupling_info)

        # Print progress for every coupling
        print(f"  Coupling {self.coupling_count:3d}: "
              f"Token H={token_entropy:.2f}, "
              f"Byte H={byte_entropy:.2f}")

        self.coupling_count += 1

    def save_logs(self, filename: str, metadata: dict = None):
        """Save all logged coupling data."""

        # Calculate summary
        if self.couplings:
            token_entropies = [c['token_dist']['entropy'] for c in self.couplings]
            byte_entropies = [c['byte_dist']['entropy'] for c in self.couplings]

            summary = {
                'total_couplings': len(self.couplings),
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
                'peaked_tokens_count': sum(1 for c in self.couplings
                                          if c['token_dist']['peaked']),
                'avg_info_per_token': round(float(np.mean(token_entropies)), 4),
            }
        else:
            summary = {}

        # Build log data
        log_data = {
            'metadata': metadata or {},
            'couplings': self.couplings,
            'summary': summary,
        }

        # Save to file
        log_file = self.log_dir / filename
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"COUPLING MATRIX LOGGING COMPLETE")
        print(f"{'='*70}")
        if summary:
            print(f"Total matrices logged: {summary['total_couplings']}")
            print(f"Average token entropy: {summary['token_entropy']['mean']:.2f} bits")
            print(f"Average byte entropy: {summary['byte_entropy']['mean']:.2f} bits")
            print(f"Peaked tokens: {summary['peaked_tokens_count']}/{summary['total_couplings']}")

        print(f"\n✓ Saved to: {log_file}")
        print(f"{'='*70}\n")

        return log_file

    def unpatch(self):
        """Restore original greedy_mec."""
        if self._is_patched and self._original_greedy_mec:
            import mec.iterative.algorithms as mec_iter
            mec_iter.greedy_mec = self._original_greedy_mec
            self._is_patched = False
            print("✓ Restored original greedy_mec")
