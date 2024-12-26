import os


tsxs_files_names = ["Поздравление_1.txt", 
                    "Поздравление_2.txt",
                    "Поздравление_3.txt",
                    "Поздравление_4.txt",
                    "Поздравление_5.txt",
                    "prompt1_арбитражка.txt"]

for fn in tsxs_files_names:
    with open(os.path.join("data", "prompt1_арбитражка.txt"), "r") as f:
        tx_prm = f.read()
    print(tx_prm)