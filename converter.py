import os

# the file that you want to convert
path = r"C:\Users\reesa\Desktop\TESTINGIMG\3D60\splits\3dv19\test_copy.txt"
# the name of the converted file
output_path = r".\new_test.txt"

if __name__ == "__main__":
    path.replace(os.sep, '/')
    newLines = []
    with open(path) as fp:
        Lines = fp.readlines()

        for line in Lines:
            print(line)
            newLines.append(line.replace('\\', "/"))

    with open(output_path, "w") as fp:
        for line in newLines:
            fp.write(line)
