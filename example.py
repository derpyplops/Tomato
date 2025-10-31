from tomato import Encoder

print('Creating encoder with GPT-2...')
encoder = Encoder(model_name='gpt2', prompt="""Q: How do you peel an apple?
A: """)

print('Encoding message...')
plaintext = 'Help'
formatted_stegotext, stegotext = encoder.encode(plaintext)

print('Decoding message...')
estimated_plaintext, estimated_bytetext = encoder.decode(stegotext)

print('\nResults:')
print('Stegotext (covertext):')
print(formatted_stegotext)
print('\n------')
print('Decoded plaintext:')
print(estimated_plaintext)
