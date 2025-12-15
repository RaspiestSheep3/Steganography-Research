import random


def PickVar(swing, amount):
    checks = []
    
    for i in range(amount):
        checks.append(random.randint(0,1))
        
    return (swing if(swing in checks) else checks[0])

#Throwing together a basic model
plaintext = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

plaintextCount = [0] * 9
xorCount = [0] * 9

for i in range(1024):
    random.seed(i)

    swings = [None] * len(plaintext)

    counter = 0
    for character in plaintext:
        binary = (format(((ord(character) + random.randint(0,256)) % 256), "08b"))
        
        favouredCharacter = 1 if binary.count("1") > binary.count("0") else 0

        swings[counter] = 0 if favouredCharacter == 1 else 1

        out = ""

        for bit in binary:
            bit = int(bit)
            
            bitNew = (bit ^ PickVar(0 if favouredCharacter == 1 else 1, 16))
             
            out += str(bitNew)
        
        #print(f"character : {binary}, out : {out} | counts : {binary.count("1")}, {out.count("1")}")
        
        plaintextCount[binary.count("1")] += 1
        xorCount[out.count("1")] += 1
        
        counter += 1

print(plaintextCount, xorCount)