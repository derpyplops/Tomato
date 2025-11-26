import re
import secrets
from tomato.utils.random_string import RandomString
from tomato.utils.model_marginal import ModelMarginal
from tomato.utils.debug_logger import FIMECDebugLogger
from mec import FIMEC
import numpy as np
from typing import Tuple, Optional, Dict, List

class Encoder:
    """
    This class implements encrypted steganography using FIMEC (a mechanism
    for generating covertext that is statistically indistinguishable from
    innocuous content).
    """

    def __init__(
        self,
        cipher_len: int = 15,
        shared_private_key: Optional[bytes] = None,
        prompt: str = "Good evening.",
        max_len: int = 100,
        temperature: float = 1.0,
        k: int = 50,
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    ) -> None:
        """
        Initializes the Encoder with the necessary parameters for encrypted steganography.
        
        Args:
            cipher_len (int): Length of the cipher in bytes. Default is 15.
            shared_private_key (bytes, optional): Shared private key for encryption. 
                If None, a random key is generated. Default is None.
            prompt (str): Prompt for the covertext model. Default is "Good evening."
            max_len (int): Maximum length of the covertext. Default is 100.
            temperature (float): Sampling temperature for the covertext model. Default is 1.0.
            k (int): The top-k sampling parameter for the covertext model. Default is 50.
            model_name (str): Name of the language model used for generating covertext. 
                Default is "unsloth/mistral-7b-instruct-v0.3-bnb-4bit".
        """

        # For encrypted steganography, the sender and receiver share a private key.
        self._cipher_len = cipher_len
        self._shared_private_key = shared_private_key or secrets.token_bytes(cipher_len)
        self._prompt = prompt
        self._max_len = max_len
        self._temperature = temperature
        self._model_name = model_name
        self._k = k

        # The covertext distribution is a distribution over innocuous content.
        self._covertext_dist = ModelMarginal(
            prompt=self._prompt,
            max_len=self._max_len,
            temperature=self._temperature,
            k=self._k,
            model_name=self._model_name
        )

        # Ciphertext distribution (uniform random string)
        ciphertext_dist = RandomString(num_chars=2**8, string_len=self._cipher_len)
        
        # FIMEC defines the communication protocol between the sender and receiver.
        self._imec = FIMEC(ciphertext_dist, self._covertext_dist)

    def _calculate_failure_probabilities(self, posterior, original_ciphertext: List[int]) -> Dict:
        """
        Calculate per-byte and overall failure probabilities from posterior distributions.

        This function computes the probability that decoding will fail for each byte
        and overall, based on the posterior distributions after encoding.

        Args:
            posterior: FactoredPosterior object with component_distributions attribute
            original_ciphertext: List[int] of length cipher_len, ground truth values [0-255]

        Returns:
            dict with the following keys:
                - 'per_byte_p_correct': List[float] - P(correct) for each byte
                - 'per_byte_p_fail': List[float] - P(fail) for each byte
                - 'entropies': List[float] - Shannon entropy for each byte (bits)
                - 'p_success_overall': float - probability all bytes decode correctly
                - 'p_fail_overall': float - probability at least one byte fails
                - 'avg_entropy': float - average entropy across all bytes
                - 'num_weak_bytes': int - count of bytes with p_fail > 0.1
                - 'num_strong_bytes': int - count of bytes with p_fail < 0.01
        """
        results = {
            'per_byte_p_correct': [],
            'per_byte_p_fail': [],
            'entropies': []
        }

        # Iterate through each byte
        for i in range(len(original_ciphertext)):
            # Get posterior distribution for this byte
            posterior_i = posterior.component_distributions[i]

            # Get correct value
            correct_value = original_ciphertext[i]

            # Calculate p_correct and p_fail
            p_correct = float(posterior_i[correct_value])
            p_fail = 1.0 - p_correct

            results['per_byte_p_correct'].append(p_correct)
            results['per_byte_p_fail'].append(p_fail)

            # Calculate entropy: H(X) = -Σ p(x) * log₂(p(x))
            nonzero_probs = posterior_i[posterior_i > 0]
            if len(nonzero_probs) > 0:
                entropy = float(-(nonzero_probs * np.log2(nonzero_probs)).sum())
            else:
                entropy = 0.0
            results['entropies'].append(entropy)

        # Calculate overall metrics
        results['p_success_overall'] = float(np.prod(results['per_byte_p_correct']))
        results['p_fail_overall'] = 1.0 - results['p_success_overall']
        results['avg_entropy'] = float(np.mean(results['entropies']))
        results['num_weak_bytes'] = sum(1 for p in results['per_byte_p_fail'] if p > 0.1)
        results['num_strong_bytes'] = sum(1 for p in results['per_byte_p_fail'] if p < 0.01)

        return results

    def encode(self, plaintext: str = "Attack at dawn!", debug: bool = False, calculate_failure_probs: bool = True) -> Tuple[str, np.ndarray, Dict]:
        """
        Encodes the plaintext into stegotext using encrypted steganography.

        Optionally computes failure probabilities during encoding to predict decoding reliability.

        Args:
            plaintext (str): The message to encode. Default is "Attack at dawn!".
            debug (bool): If True, logs debug information to ./logs directory. Default is False.
            calculate_failure_probs (bool): If True, computes posterior distributions and failure
                probabilities (requires additional decoding pass). Default is True.

        Returns:
            Tuple[str, np.ndarray, Dict]: The formatted stegotext, original stegotext, and
            failure probability analysis (empty dict if calculate_failure_probs=False) containing:
                - 'per_byte_p_correct': List[float] - probability each byte decodes correctly
                - 'per_byte_p_fail': List[float] - probability each byte fails
                - 'entropies': List[float] - Shannon entropy per byte (bits)
                - 'p_success_overall': float - probability all bytes decode correctly
                - 'p_fail_overall': float - probability at least one byte fails
                - 'avg_entropy': float - average entropy across bytes
                - 'num_weak_bytes': int - count of bytes with p_fail > 0.1
                - 'num_strong_bytes': int - count of bytes with p_fail < 0.01
        """
        # Setup debug logging if requested
        logger = None
        if debug:
            logger = FIMECDebugLogger(log_dir="./logs", operation="encode")
            logger.start(
                plaintext=plaintext,
                cipher_len=self._cipher_len,
                max_len=self._max_len,
                k=self._k,
                temperature=self._temperature,
                model_name=self._model_name,
                prompt=self._prompt[:50] + "..." if len(self._prompt) > 50 else self._prompt
            )

        # Convert plaintext to a sequence of bytes.
        bytetext = plaintext.encode("utf-8")

        # Pad bytetext if necessary to match the cipher length.
        if len(bytetext) < self._cipher_len:
            bytetext += b'A' * (self._cipher_len - len(bytetext))

        if len(bytetext) != self._cipher_len:
            raise ValueError("The length of the bytetext representation of the plaintext should be less than or equal to the cipher length provided.")

        # Encrypt the plaintext with the shared private key to generate ciphertext.
        ciphertext = [a ^ b for a, b in zip(bytetext, self._shared_private_key)]

        if debug and logger:
            logger.logs['true_ciphertext'] = [int(c) for c in ciphertext]
            logger.logs['true_bytetext'] = [int(b) for b in bytetext]

        # Generate stegotext with the ciphertext hidden inside.
        stegotext, _ = self._imec.sample_y_given_x(ciphertext)

        # Optionally compute posterior distributions to calculate failure probabilities
        # This predicts decoding reliability at encoding time (requires extra decoding pass)
        if calculate_failure_probs:
            posterior = self._imec._x_given_y(stegotext)
            failure_probs = self._calculate_failure_probabilities(posterior, ciphertext)
        else:
            failure_probs = {}

        # Format the stegotext by replacing multiple spaces with newlines.
        formatted_stegotext = re.sub(" {2,}", "\n", self._covertext_dist.decode(stegotext).replace("\n", " ")).strip()

        if debug and logger:
            result_data = {
                'stegotext_length': int(len(stegotext)),
                'formatted_stegotext': formatted_stegotext,
                'stegotext_tokens': [int(t) for t in stegotext],
            }

            if calculate_failure_probs and failure_probs:
                result_data['failure_probabilities'] = {
                    'p_fail_overall': failure_probs['p_fail_overall'],
                    'p_success_overall': failure_probs['p_success_overall'],
                    'avg_entropy': failure_probs['avg_entropy'],
                    'num_weak_bytes': failure_probs['num_weak_bytes'],
                    'num_strong_bytes': failure_probs['num_strong_bytes'],
                }

            logger.log_result(result_data)
            logger.finish()

        return formatted_stegotext, stegotext, failure_probs

    def decode(self, stegotext: np.ndarray, debug: bool = False, true_plaintext: Optional[str] = None) -> Tuple[str, bytes]:
        """
        Decodes the stegotext back into plaintext.

        Args:
            stegotext (np.ndarray): The stegotext to decode.
            debug (bool): If True, logs debug information to ./logs directory. Default is False.
            true_plaintext (str, optional): The true plaintext for comparison in debug mode.

        Returns:
            Tuple[str, bytes]: The estimated plaintext and its byte representation.
        """
        # Setup debug logging if requested
        logger = None
        if debug:
            logger = FIMECDebugLogger(log_dir="./logs", operation="decode")
            logger.start(
                stegotext_length=int(len(stegotext)),
                cipher_len=self._cipher_len,
                max_len=self._max_len,
                k=self._k,
                true_plaintext=true_plaintext if true_plaintext else "unknown"
            )

        # Estimate the ciphertext from the stegotext.
        estimated_ciphertext, _ = self._imec.estimate_x_given_y(stegotext)

        # Decrypt the estimated ciphertext to retrieve the original bytetext.
        estimated_bytetext = bytes(
            [a ^ b for a, b in zip(estimated_ciphertext, self._shared_private_key)]
        )

        # Decode the bytetext back into a string.
        estimated_plaintext = estimated_bytetext.decode("utf-8", errors="replace")

        if debug and logger:
            result_data = {
                'decoded_plaintext': estimated_plaintext,
                'decoded_bytetext': [int(b) for b in estimated_bytetext],
                'decoded_ciphertext': [int(c) for c in estimated_ciphertext],
                'stegotext_tokens': [int(t) for t in stegotext],
            }

            # Add accuracy info if true plaintext provided
            if true_plaintext:
                true_bytetext = true_plaintext.encode("utf-8")
                if len(true_bytetext) < self._cipher_len:
                    true_bytetext += b'A' * (self._cipher_len - len(true_bytetext))

                true_ciphertext = [a ^ b for a, b in zip(true_bytetext, self._shared_private_key)]

                # Calculate accuracy
                correct_bytes = sum(1 for t, e in zip(true_ciphertext, estimated_ciphertext) if t == e)
                accuracy = correct_bytes / self._cipher_len

                # Find errors
                errors = []
                for i, (t, e) in enumerate(zip(true_ciphertext, estimated_ciphertext)):
                    if t != e:
                        errors.append({
                            'byte_idx': i,
                            'true_value': int(t),
                            'decoded_value': int(e),
                            'true_char': chr(true_bytetext[i]) if 32 <= true_bytetext[i] < 127 else f'\\x{true_bytetext[i]:02x}',
                            'decoded_char': chr(estimated_bytetext[i]) if 32 <= estimated_bytetext[i] < 127 else f'\\x{estimated_bytetext[i]:02x}',
                        })

                result_data.update({
                    'true_plaintext': true_plaintext,
                    'true_bytetext': [int(b) for b in true_bytetext],
                    'true_ciphertext': [int(c) for c in true_ciphertext],
                    'accuracy': round(accuracy, 4),
                    'correct_bytes': correct_bytes,
                    'total_bytes': self._cipher_len,
                    'errors': errors,
                })

            logger.log_result(result_data)
            logger.finish()

        return estimated_plaintext, estimated_bytetext
