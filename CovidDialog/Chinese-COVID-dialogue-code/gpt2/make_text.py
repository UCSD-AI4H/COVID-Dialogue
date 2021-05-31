import json

from tqdm import tqdm

def get_data(file_name):
    f_in = open(file_name)
    data = json.load(f_in)
    f_in.close()
    new_name = file_name.replace(".json", ".txt").replace("data", "med")
    f_out = open("data/" + new_name, "w")
    total = 1
    for dialogs in tqdm(data):
        for utts in dialogs:
            utt = utts[3:]
            f_out.write(utt + "\n")
        if (total < len(data)):
            f_out.write("\n")
        total += 1
    f_out.close()
    return total - 1
    

if __name__ == "__main__":
    total_train = get_data("../data/train_data.json")
    print ("total_train: ", total_train)
    total_test = get_data("../data/test_data.json")
    print ("total_test: ", total_test)
    total_valid = get_data("../data/validate_data.json")
    print ("total_validate: ", total_valid)
    
        
        

