import numpy as np
from numpy import matrix
from numpy import linalg
import re
import time

def modMatInv(A,p):
  n=len(A)
  A=matrix(A)
  adj=np.zeros(shape=(n,n))

  for i in range(0,n):
    for j in range(0,n):
      adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
  return (modInv(int(round(linalg.det(A))),p)*adj)%p

def modInv(a,p):
  for i in range(1,p):
    if (i*a)%p==1:
      return i
  raise ValueError(str(a)+" has no inverse mod "+str(p))

def minor(A,i,j):
  A=np.array(A)
  minor=np.zeros(shape=(len(A)-1,len(A)-1))
  p=0
  for s in range(0,len(minor)):
    if p==i:
      p=p+1
    q=0
    for t in range(0,len(minor)):
      if q==j:
        q=q+1
      minor[s][t]=A[p][q]
      q=q+1
    p=p+1
  return minor


def generate_random_matrix_with_gcd_det(k, n):
  matrix = np.zeros((k, k), dtype=int)
  while True:
    for i in range(k):
      for j in range(k):
        num = np.random.randint(0, n)
        matrix[i][j] = num

    if np.gcd(int(round(np.linalg.det(matrix))), n) == 1:
      break

  return matrix

def generate_s_key(lenofcipherkeymat,mod=None):
  if mod is None:
    vectoradd = np.random.randint(0, 32, size=(lenofcipherkeymat, 1))
  else:
    vectoradd = np.random.randint(0, 25, size=(lenofcipherkeymat, 1))
  return vectoradd

def totext(matrix, alphabet=None):
    if alphabet is None:
      # Default to Ukrainian alphabet
      alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    elif alphabet == "EN":
      # Use English alphabet
      alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    else:
      raise ValueError("Unsupported alphabet")

    finalmessage = [alphabet[l] for l in np.ravel(matrix.transpose())]

    return "".join(finalmessage)


"""def toindexv2(plaintext, num_rows, alphabet=None):
  if alphabet is None:
    # Default to Ukrainian alphabet
    alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    plaintext = re.sub('[^А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]+', '', plaintext).upper()
  elif alphabet == "EN":
    # Use English alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plaintext = re.sub('[^a-zA-Z]+', '', plaintext).upper()
  else:
    raise ValueError("Unsupported alphabet")
  indexed = []
  for char in plaintext:  # Loop through each char in plaintext.
    num = alphabet.find(char)
    num %= len(alphabet)
    indexed.append(num)
  #print(len(indexed))
  if(len(indexed)%num_rows==0):
    num_cols = len(indexed) // num_rows  # Calculate the number of columns based on the number of rows
    indexed = np.reshape(indexed, (num_cols, num_rows)).transpose()  # Reshape and transpose the matrix
  else:
    while True:
      if alphabet is None:
        indexed.append(np.random.randint(0,32))
      else:
        indexed.append(np.random.randint(0,25))
      if(len(indexed)%num_rows==0):
        num_cols = len(indexed) // num_rows  # Calculate the number of columns based on the number of rows
        indexed = np.reshape(indexed, (num_cols, num_rows)).transpose()  # Reshape and transpose the matrix
        break

  return indexed"""

def toindexv3(plaintext, num_rows, alphabet=None):
  if alphabet is None:
    # Default to Ukrainian alphabet
    alphabet = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    plaintext = re.sub('[^А-ЩЬЮЯЄІЇҐа-щьюяєіїґ]+', '', plaintext).upper()
  elif alphabet == "EN":
    # Use English alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plaintext = re.sub('[^a-zA-Z]+', '', plaintext).upper()
  else:
    raise ValueError("Unsupported alphabet")
  indexed = [alphabet.find(char) % len(alphabet) for char in plaintext]
  #print(len(indexed))
  if(len(indexed)%num_rows==0):
    num_cols = len(indexed) // num_rows
    indexed = np.reshape(indexed, (num_cols, num_rows)).transpose()
  else:
    while True:
      if alphabet is None:
        indexed.append(np.random.randint(0,32))
      else:
        indexed.append(np.random.randint(0,25))
      if(len(indexed)%num_rows==0):
        num_cols = len(indexed) // num_rows
        indexed = np.reshape(indexed, (num_cols, num_rows)).transpose()
        break

  return indexed


def readkeysfromfile(filepath, filepath1, mod=33):
  with open(filepath, 'r') as f:
    lines = f.readlines()
  matrix = []

  for line in lines:

    line_values = line.strip().split()
    values = []

    for val in line_values:
      val = int(val)
      values.append(val)

    matrix.append(values)

  matrix = np.matrix(matrix).astype(int)%mod
  # print(matrix)
  #print(matrix)
  #print(int(round(np.linalg.det(matrix))))
  gcddet = np.gcd(int(round(np.linalg.det(matrix))), mod)

  if gcddet == 1:

    with open(filepath1, 'r') as f1:
      lines1 = f1.readlines()

    listvec = []
    for line in lines1:
      listvec.append([int(line)%mod])

    if len(listvec) == len(matrix):
      return matrix, np.reshape(listvec, (-1, 1))
    else:
      raise ValueError(f"Length of vector: {len(listvec)} is not the same as matrix length: {len(matrix)}")
  else:
    raise ValueError(f"Matrix determinant has GCD {gcddet}, not 1")

"""def encryptv2(message,keymatrix,mod=33,alphabet=None):
  matrixofmessage=toindexv2(message,len(keymatrix),alphabet)
  multipliedmatrixandkey=np.dot(keymatrix,matrixofmessage)%mod
  return totext(multipliedmatrixandkey)
  
def decryptv2(ciphertext,cipherkeymatrix,mod=33,alphabet=None):
  matrixcipher=toindexv2(ciphertext,len(cipherkeymatrix))
  #print(cipherkeymatrix)
  inversekeymatrix=modMatInv(cipherkeymatrix,mod)
  decryptmatrix=np.dot(inversekeymatrix,matrixcipher).astype(int)%mod
  return totext(decryptmatrix,alphabet)"""

def encryptv3(message,keymatrix,vectorkey,mod=33,alphabet=None):
  global stenctime, stopenctime
  stenctime = time.time()
  matrixofmessage=toindexv3(message,len(keymatrix),alphabet)
  multipliedmatrixandkey=np.dot(keymatrix,matrixofmessage)%mod

  #print(multipliedmatrixandkey)
  finalmatrix=(multipliedmatrixandkey+vectorkey)%mod
  #print(finalmatrix)
  stopenctime=time.time()
  return totext(finalmatrix,alphabet)

def decryptv3(ciphertext,cipherkeymatrix,ciphervectorkey,mod=33,alphabet=None):
  global stdectime,stopdectime
  stdectime=time.time()
  matrixcipher=toindexv3(ciphertext,len(cipherkeymatrix),alphabet)
  #print(cipherkeymatrix)
  inversekeymatrix=modMatInv(cipherkeymatrix,mod)
  decryptmatrix=np.dot(inversekeymatrix,matrixcipher).astype(int)%mod
  #print(decryptmatrix)
  #print(ciphervectorkey)
  inversevector=(-1) * np.dot(inversekeymatrix, ciphervectorkey).astype(int) % mod
  #print(inversevector)
  finalmatrix=(decryptmatrix+inversevector)%mod
  #print(finalmatrix)
  stopdectime=time.time()
  return totext(finalmatrix,alphabet)


def encryptformenu(mesgin=None,modp=33,alphabetp=None):
  global keymatrix1, vectorkey1, genkeymat, genvectkey
  if mesgin == "file":
    filetoreadpath = input("Type in name of the txt file, or path to it: ")
    if alphabetp is None:
      with open(filetoreadpath, 'r') as f1:
        textfromfile = f1.read()
    else:
      with open(filetoreadpath, 'r',encoding="utf-8") as f1:
        textfromfile = f1.read()
    keysfrom = input("If you want to ENCRYPT with your keys, press 'file',\nif you want to generate keys, press anything else: ")
    if keysfrom == "file":
      matkeypath = input("Type in name of the txt file where is key matrix is, or path to it: ")
      vectkeypath = input("Type in name of the txt file where is key matrix is, or path to it: ")
      keymat, keyvect = readkeysfromfile(matkeypath, vectkeypath,modp)
      if alphabetp==None:
        encryptedfromfile = encryptv3(textfromfile, keymat, keyvect)
      else:
        encryptedfromfile = encryptv3(textfromfile, keymat, keyvect,mod=modp,alphabet=alphabetp)
      with open("encaph.txt", "w") as fileout:
        fileout.write(encryptedfromfile)
      print(encryptedfromfile)
    else:
      lenofgenkeymatrix = int(input("Enter the size of key matrix: "))
      if input("If you want to export keys in files type in yes: ")=="yes":
        genkeymat = generate_random_matrix_with_gcd_det(lenofgenkeymatrix, modp)
        genvectkey = generate_s_key(len(genkeymat),modp)
        with open("exportkeymatrix.txt",'w') as mex,open("exportkeyvector.txt",'w') as vex:
          mex.write(re.sub(r'[\[\]]', '',str(genkeymat)))
          vex.write(re.sub(r'[\[\]]', '',str(genvectkey)))
      else:
        genkeymat = generate_random_matrix_with_gcd_det(lenofgenkeymatrix, modp)
        genvectkey = generate_s_key(len(genkeymat), modp)
      if alphabetp == None:
        encryptedfromfilewithgenkeys = encryptv3(textfromfile, genkeymat, genvectkey)
      else:
        print("hello")
        encryptedfromfilewithgenkeys = encryptv3(textfromfile, genkeymat, genvectkey, mod=modp, alphabet=alphabetp)
      with open("encaph.txt", "w") as fileout:
        fileout.write(encryptedfromfilewithgenkeys)
      print("Ciphertext: ", encryptedfromfilewithgenkeys)
  else:
    message = input("Enter the message you want to encrypt: ")

    lenofkeymatrix = int(input("Enter the size of key matrix: "))
    keymatrix1 = generate_random_matrix_with_gcd_det(lenofkeymatrix, modp)
    vectorkey1 = generate_s_key(len(keymatrix1),modp)

    if alphabetp == None:
      ciphertext = encryptv3(message, keymatrix1, vectorkey1)
    else:
      #print("hello")
      ciphertext = encryptv3(message, keymatrix1, vectorkey1, mod=modp, alphabet=alphabetp)
    print("Ciphertext:", ciphertext)
  print(f'Time for decryption: {stopenctime - stenctime}')




def decryptformenu(mesgin=None,modp=33,alphabetp=None):
  if mesgin == "file":
    filetoreadpath = input("Type in name of the txt file, or path to it: ")
    """if alphabetp is None:
      with open(filetoreadpath, 'r') as f1:
        textfromfile = f1.read()
    else:
      with open(filetoreadpath, 'r', encoding="utf-8") as f1:
        textfromfile = f1.read()"""
    with open(filetoreadpath, 'r') as f1:
      textfromfile = f1.read()
    #print(textfromfile)
    keysfrom = input("If you want to DECRYPT with your keys, press 'file',\nif you want to generate keys, press anything else: ")
    if keysfrom == "file":
      matkeypath = input("Type in name of the txt file where is key matrix is, or path to it: ")
      vectkeypath = input("Type in name of the txt file where is key matrix is, or path to it: ")
      keymat, keyvect = readkeysfromfile(matkeypath, vectkeypath,modp)
      if alphabetp==None:
        decryptedfromfile = decryptv3(textfromfile, keymat, keyvect)
      else:
        decryptedfromfile = decryptv3(textfromfile, keymat, keyvect,mod=modp,alphabet=alphabetp)
      with open("decrypth.txt", "w") as fileout:
        fileout.write(decryptedfromfile)
      print("Plaintext: ", decryptedfromfile)
    else:
      genkeymatdec = genkeymat
      genvectkeydec = genvectkey
      if alphabetp == None:
        decryptedfromfilewithgenkeys = decryptv3(textfromfile, genkeymatdec, genvectkeydec)
      else:
        decryptedfromfilewithgenkeys = decryptv3(textfromfile, genkeymatdec, genvectkeydec, mod=modp, alphabet=alphabetp)
      with open("decrypth.txt", "w") as fileout:
        fileout.write(decryptedfromfilewithgenkeys)
      print("Plaintext: ", decryptedfromfilewithgenkeys)
  else:
    message = input("Enter the message you want to decrypt: ")
    keymatrixdec = keymatrix1
    vectorkeydec = vectorkey1
    if alphabetp == None:
      plaintext = decryptv3(message, keymatrixdec, vectorkeydec)
    else:
      plaintext = decryptv3(message, keymatrixdec, vectorkeydec, mod=modp, alphabet=alphabetp)
    print("Plaintext:", plaintext)
  print(f'Time for decryption: {stopdectime - stdectime}')

def encrypt_file_by_blocks(input_file_path, output_file_path, keymatrix, vectorkey, block_size=1024,alphabet=None,mod=33):
  with open(input_file_path, 'r',encoding='utf-8') as input_file, open(output_file_path, 'w') as output_file:
    while True:
      block = input_file.read(block_size)
      if not block:
        break
      encrypted_block = encryptv3(block, keymatrix, vectorkey,mod=mod,alphabet=alphabet)
      #print(encrypted_block)
      output_file.write(encrypted_block)

def decrypt_file_by_blocks(input_file_path, output_file_path, keymatrix, vectorkey, block_size=1024,alphabet=None,mod=33):
  with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    while True:
      block = input_file.read(block_size)
      if not block:
        break
      decrypted_block = decryptv3(block, keymatrix, vectorkey,mod=mod,alphabet=alphabet)
      #print(encrypted_block)
      output_file.write(decrypted_block)
def main():
  """
    with open("pg100.txt",'r',encoding='utf-8') as textread:
      readedtext=textread.read()



    #with open("notbyblocksappend.txt", 'w') as textread1:
     # textread1.write()


    start_time = time.time()
    abet = encryptv3(readedtext, genkeymat, genkeyvec, mod=26, alphabet="EN")
    stop_time = time.time()
    print(stop_time - start_time)

    st = time.time()
    decryptv3(abet, genkeymat, genkeyvec, mod=26, alphabet="EN")
    stop = time.time()
    print(stop - st)"""

  """genkeymatblock=generate_random_matrix_with_gcd_det(4,26)
  genkeyvecblock=generate_s_key(len(genkeymatblock))
  start1=time.time()
  encrypt_file_by_blocks("pg100.txt","byblocks.txt",genkeymatblock,genkeyvecblock,mod=26,alphabet="EN")
  stop2 = time.time()
  print(stop2-start1)
  st3 = time.time()
  decrypt_file_by_blocks("byblocks.txt", "byblocksdec1.txt", genkeymatblock, genkeyvecblock,mod=26,alphabet="EN")
  stop3 = time.time()
  print(stop3 - st3)"""
  testmatrixkey=generate_random_matrix_with_gcd_det(2,26)
  testvectkey=generate_s_key(len(testmatrixkey),26)
  texts="""This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook."""
  matrix=[[1,1],[32,1]]
  vector=[[1],[15]]
  text1='завтразавтразавтра'

  print(encryptv3(text1,matrix,vector))
  tryenc=encryptv3(texts,testmatrixkey,testvectkey,mod=26,alphabet="EN")

  print(tryenc)


  while True:
    print("1. Encrypt a message")
    print("2. Decrypt a message")
    print("3. Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
      paramalphabet=input("If you want to use english alphabet to encrypt or decrypt, type in 'ENG',\nif you want to use Ukranian, type in anything: ")
      mesgin = input("\nIf you want to ENCRYPT text from file type in  'file' ,\nif you want to type in and ENCRYPT it, press anything else: ")
      if paramalphabet=="ENG":
        if mesgin == "file":
          encryptformenu(mesgin="file",modp=26,alphabetp="EN")
        else:
          encryptformenu(modp=26, alphabetp="EN")
      else:
        if mesgin=="file":
          encryptformenu(mesgin="file")
        else:
          encryptformenu()
    elif choice == "2":
      paramalphabet = input("If you want to use english alphabet to encrypt or decrypt, type in 'ENG',\nif you want to use Ukranian, type in anything: ")
      mesgin = input("\nIf you want to ENCRYPT text from file type in  'file' ,\nif you want to type in and ENCRYPT it, press anything else: ")
      if paramalphabet == "ENG":
        if mesgin == "file":
          decryptformenu(mesgin="file", modp=26, alphabetp="EN")
        else:
          decryptformenu(modp=26, alphabetp="EN")
      else:
        if mesgin == "file":
          decryptformenu(mesgin="file")
        else:
          decryptformenu()
    elif choice == "3":
      print("Exiting the program...")
      break
    else:
      print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()















