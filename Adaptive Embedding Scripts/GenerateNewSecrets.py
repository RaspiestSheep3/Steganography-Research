import os
import random
from Helpers.HelperFunctions import GetPaths
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

"""
- This code generates the random binary and encrypted payload used when comparing differences to the composite method.
- The structured text was generated from https://www.lipsum.com/
- The code is not cryptographically secure nor is it meant to be - it just provides some example ciphertext for comparison
- Using AES-256 as this is what is used in TLS 1.3
"""

SETUP_PATHS = GetPaths()
LIPSUM = SETUP_PATHS["Structured Secret"]
BINARY_SECRET = SETUP_PATHS["Binary Secret"]
ENCRYPTED_SECRET = SETUP_PATHS["Encrypted Secret"]

SEED = 12345

def GenerateBinary(binaryLength, randomSeed):
    random.seed(randomSeed)

    with open(BINARY_SECRET, "wb") as f:
        f.write(random.randbytes(binaryLength))

def GenerateEncryptedSecret(randomSeed):
    random.seed(randomSeed)
    
    aesKey = random.randbytes(32)
    aes = AESGCM(aesKey)
    nonce = random.randbytes(12)

    with open(LIPSUM, "r") as f:
        ciphertext = aes.encrypt(
            nonce,
            f.read().encode(),
            None
        )

    with open(ENCRYPTED_SECRET, "wb") as f:
        f.write(ciphertext)

if __name__ == "__main__":
    generateBinary = input("Generate Binary (Y/N) : ").upper() == "Y"
    generateEncrypted = input("Generate Ciphertext (Y/N) : ").upper() == "Y"
    length = int((256 * 256 // 8) * 1.5) #Max secret for LSB embedding * 1.5 just in case

    if(generateBinary):
        GenerateBinary(length, SEED)

    if(generateEncrypted):
        GenerateEncryptedSecret(SEED)