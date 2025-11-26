import re
import secrets
from tomato.utils.random_string import RandomString
from tomato.utils.model_marginal import ModelMarginal
from tomato.utils.debug_logger import FIMECDebugLogger
from mec import FIMEC
import numpy as np
from typing import Tuple, Optional, Dict, List, Generator, Union

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

    def encode_stream(
        self,
        plaintext: str = "Attack at dawn!",
        chunk_size: int = 1,
        calculate_failure_probs: bool = False
    ) -> Generator[Dict[str, Union[str, int, bool]], None, None]:
        """
        Encodes plaintext into stegotext and yields tokens as they're generated in real-time.

        This method implements TRUE streaming by monkey-patching FIMEC's _main_loop
        to yield each token as it's generated by the underlying algorithm.

        Args:
            plaintext (str): The message to encode. Default is "Attack at dawn!".
            chunk_size (int): Number of tokens to yield per chunk. Default is 1 (true streaming).
            calculate_failure_probs (bool): If True, computes failure probabilities. Default is False.

        Yields:
            Dict containing:
                - 'type': 'token' | 'complete'
                - 'text': Decoded text for this token/chunk (for 'token' type)
                - 'token_id': Token ID (for 'token' type with chunk_size=1)
                - 'tokens': Token IDs for this chunk (for 'token' type with chunk_size>1)
                - 'token_index': Current token index (for 'token' type)
                - 'total_tokens': Total number of tokens (for 'complete' type)
                - 'failure_probs': Failure probability dict (for 'complete' type if enabled)
        """
        # Convert plaintext to bytes
        bytetext = plaintext.encode("utf-8")

        # Pad if necessary
        if len(bytetext) < self._cipher_len:
            bytetext += b'A' * (self._cipher_len - len(bytetext))

        if len(bytetext) != self._cipher_len:
            raise ValueError("Plaintext too long for cipher length")

        # Encrypt
        ciphertext = [a ^ b for a, b in zip(bytetext, self._shared_private_key)]

        # Create streaming generator that yields tokens in real-time
        def generate_with_streaming():
            """Generator that yields tokens as FIMEC produces them"""
            # Prepare for encoding
            helper = self._imec._make_helper(self._imec._select_sample_y_j, x=ciphertext, y=None)

            # Initialize FIMEC state
            posterior = self._imec._initialize_posterior()
            y_prefix = []
            likelihoods = []
            chunk_buffer = []

            # Generate tokens one by one
            while not self._imec.nu.is_terminal(y_prefix):
                y_j_conditional = self._imec._nu_conditional(y_prefix)
                y_j, y_j_likelihood, posterior = self._imec._iterate(
                    helper, posterior, y_prefix, y_j_conditional
                )
                y_prefix.append(y_j)
                likelihoods.append(y_j_likelihood)

                # Yield token immediately as it's generated!
                chunk_buffer.append(y_j)

                if len(chunk_buffer) >= chunk_size:
                    # Decode ALL tokens so far for proper accumulated text display
                    accumulated_text = self._covertext_dist.decode(y_prefix)

                    if chunk_size == 1:
                        yield {
                            'type': 'token',
                            'text': accumulated_text,
                            'token_id': int(chunk_buffer[0]),
                            'token_index': len(y_prefix) - 1
                        }
                    else:
                        yield {
                            'type': 'token',
                            'text': accumulated_text,
                            'tokens': [int(t) for t in chunk_buffer],
                            'token_index': len(y_prefix) - len(chunk_buffer),
                            'chunk_size': len(chunk_buffer)
                        }

                    chunk_buffer = []

            # Yield any remaining tokens
            if chunk_buffer:
                # Decode ALL tokens so far for proper accumulated text display
                accumulated_text = self._covertext_dist.decode(y_prefix)
                if chunk_size == 1:
                    for i, token_id in enumerate(chunk_buffer):
                        yield {
                            'type': 'token',
                            'text': accumulated_text,
                            'token_id': int(token_id),
                            'token_index': len(y_prefix) - len(chunk_buffer) + i
                        }
                else:
                    yield {
                        'type': 'token',
                        'text': accumulated_text,
                        'tokens': [int(t) for t in chunk_buffer],
                        'token_index': len(y_prefix) - len(chunk_buffer),
                        'chunk_size': len(chunk_buffer)
                    }

            # Optionally calculate failure probabilities
            failure_probs = {}
            if calculate_failure_probs:
                failure_probs = self._calculate_failure_probabilities(posterior, ciphertext)

            # Format the complete stegotext
            formatted_stegotext = re.sub(
                " {2,}", "\n",
                self._covertext_dist.decode(y_prefix).replace("\n", " ")
            ).strip()

            # Yield completion message
            yield {
                'type': 'complete',
                'total_tokens': int(len(y_prefix)),
                'formatted_stegotext': formatted_stegotext,
                'stegotext': [int(t) for t in y_prefix],
                'failure_probs': failure_probs if calculate_failure_probs else {}
            }

        # Yield from the streaming generator
        yield from generate_with_streaming()

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

    def decode_stream(self, stegotext: Union[list, np.ndarray], chunk_size: int = 1) -> Generator[Dict, None, None]:
        """
        Stream the decoding process, showing the evolving "best guess" plaintext as each token is processed.

        This visualizes Bayesian inference in action - as more tokens provide evidence,
        the posterior distributions peak around the correct byte values.

        Args:
            stegotext: Token IDs of the stegotext to decode
            chunk_size: Number of tokens to process before yielding an update

        Yields:
            Dict with:
                - 'type': 'token' | 'complete'
                - 'current_guess': Current best-guess plaintext (for 'token' type)
                - 'tokens_processed': Number of tokens processed so far (for 'token' type)
                - 'confidence': Average max probability across all byte distributions (for 'token' type)
                - 'plaintext': Final decoded plaintext (for 'complete' type)
        """
        # Convert to list if needed
        if isinstance(stegotext, np.ndarray):
            stegotext = stegotext.tolist()

        def generate_with_streaming():
            """Generator that yields current guess after processing each token"""
            # Initialize posterior for decoding (start with prior)
            posterior = self._imec._initialize_posterior()
            y_prefix = []

            # Helper for decoding: we know the true y values (the stegotext)
            # Use a lambda to create a selector that returns the known token values
            def select_known_y_j(y_j_posterior, y_prefix, y=None):
                """Select the known y value at current position"""
                return stegotext[len(y_prefix)]

            helper = self._imec._make_helper(
                select_known_y_j,
                x=None,
                y=stegotext
            )

            token_buffer = []

            # Process each token one by one
            for token_idx, true_token in enumerate(stegotext):
                # Check if this prefix is valid
                if self._imec.nu.is_terminal(y_prefix):
                    break

                # Get conditional distribution for next token
                y_j_conditional = self._imec._nu_conditional(y_prefix)

                # Iterate: update posterior based on observing this token
                y_j, y_j_likelihood, posterior = self._imec._iterate(
                    helper, posterior, y_prefix, y_j_conditional
                )

                y_prefix.append(y_j)
                token_buffer.append(y_j)

                # After processing chunk_size tokens, yield current best guess
                if len(token_buffer) >= chunk_size:
                    # Extract current best guess by taking argmax of each byte's posterior
                    current_ciphertext_guess = []
                    confidences = []

                    for component_dist in posterior.component_distributions:
                        best_byte = int(component_dist.argmax())
                        confidence = float(component_dist.max())
                        current_ciphertext_guess.append(best_byte)
                        confidences.append(confidence)

                    # Decrypt the current guess
                    current_bytetext_guess = bytes(
                        [a ^ b for a, b in zip(current_ciphertext_guess, self._shared_private_key)]
                    )

                    # Decode to string
                    current_plaintext_guess = current_bytetext_guess.decode("utf-8", errors="replace")

                    # Calculate average confidence
                    avg_confidence = float(np.mean(confidences))

                    yield {
                        'type': 'token',
                        'current_guess': current_plaintext_guess,
                        'tokens_processed': len(y_prefix),
                        'total_tokens': len(stegotext),
                        'confidence': round(avg_confidence, 4),
                        'byte_confidences': [round(c, 4) for c in confidences]
                    }

                    token_buffer = []

            # Yield any remaining tokens
            if token_buffer:
                # Extract final best guess
                current_ciphertext_guess = []
                confidences = []

                for component_dist in posterior.component_distributions:
                    best_byte = int(component_dist.argmax())
                    confidence = float(component_dist.max())
                    current_ciphertext_guess.append(best_byte)
                    confidences.append(confidence)

                current_bytetext_guess = bytes(
                    [a ^ b for a, b in zip(current_ciphertext_guess, self._shared_private_key)]
                )
                current_plaintext_guess = current_bytetext_guess.decode("utf-8", errors="replace")
                avg_confidence = float(np.mean(confidences))

                yield {
                    'type': 'token',
                    'current_guess': current_plaintext_guess,
                    'tokens_processed': len(y_prefix),
                    'total_tokens': len(stegotext),
                    'confidence': round(avg_confidence, 4),
                    'byte_confidences': [round(c, 4) for c in confidences]
                }

            # Final decode: use the completed posterior
            final_ciphertext = []
            for component_dist in posterior.component_distributions:
                best_byte = int(component_dist.argmax())
                final_ciphertext.append(best_byte)

            final_bytetext = bytes(
                [a ^ b for a, b in zip(final_ciphertext, self._shared_private_key)]
            )
            final_plaintext = final_bytetext.decode("utf-8", errors="replace").rstrip('A')

            yield {
                'type': 'complete',
                'plaintext': final_plaintext,
                'total_tokens': len(stegotext)
            }

        yield from generate_with_streaming()
