# coding: utf-8
# Author: kaifa.zhao@connect.polyu.hk
# Copyright 2021@Kaifa Zhao (Zachary)
# Date: 2022/7/5
# System: linux
import os.path
import sys
sys.path.append("../")
import shutil
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import ppdeep
import json
from tqdm import tqdm
import subprocess
import time
from scipy.sparse import coo_matrix
from Utils.SmaliUtil import smaliFucSig2Java


framework = ["java.", "sun.", "android.", "org.apache.", "org.eclipse.", "soot.", "javax.",
            "com.google.", "org.xml.", "junit.", "org.json.", "org.w3c.com","FLOWDROID_EXCEPTIONS"]

smali_type = {"B": "byte", "I": "int", "C": "char", "D": "double","F": "float",
             "J": "long", "S": "short", "V": "void", "Z": "boolean"}


def smali2soot(smali_method):

    smali_method_str = smali_method.strip().split(" ")[-1]
    smali_method_name = smali_method_str.split("(")[0]
    return_va = smali_method_str.split(")")[-1]
    if len(return_va) == 1:
        return_str = smali_type[return_va]
    else:
        return_str = ".".join(return_va[1:-1].split("/"))
    params = smali_method_str.split("(")[1].split(")")[0].split(";")
    paras_list = []
    for par in params:
        if par == "":
            continue
        if len(par) == 1:
            par = smali_type[par]
        else:
            par = ".".join(par[1:].split("/"))
        paras_list.append(par)
    paras_str = ",".join(paras_list)
    smali2soot_str = return_str + " " + smali_method_name + "(" + paras_str + ")"
    return smali2soot_str


def analyzeGivenSmaliFile(explain_sig,samli_file):

    f = open(samli_file, "r")
    class_body = f.readlines()
    class_name = class_body[0].split(" ")[-1]
    instruction_sequence = []
    flag = False
    f.close()

    for stmt in class_body[1:]:
        if stmt.startswith(".method"):
            method_name = ".".join(stmt.split(" ")[1:])
            method_sig = class_name.replace('\n', '') + method_name.replace('\n', '')
            smali2soot_str = smaliFucSig2Java(method_sig)
            if not explain_sig==smali2soot_str:
                continue
            method_name = ".".join(stmt.split(" ")[1:])
            instruction_sequence.append('\n' + class_name.replace('\n', '') + method_name)
            flag = True
        elif stmt.startswith(".end method"):
            flag = False
        else:
            if flag and stmt != "\n":  # in method body
                if not stmt.strip().startswith("."):
                    instruction_sequence.append("\t" + stmt.strip().split(" ")[0])
            else:
                continue
    return instruction_sequence


def findSmalimethods(test_nodes,cur_path):
    instruction_seq = {}
    count = 0
    for sig in test_nodes:
        smali_path = cur_path

        package = sig[1:].split(":")[0].split(".")
        while (len(package)>1):
            itm = package.pop(0)
            smali_path = os.path.join(smali_path, itm)
        smali_path = os.path.join(smali_path, package[0]+".smali")
        if not os.path.exists(smali_path):
            continue
        instruction_sequence = analyzeGivenSmaliFile(sig, smali_path)
        instruction_seq[sig] = instruction_sequence
        if instruction_sequence==[]:
            count +=1
    print("the number of empty count is: ",str(count))
    return instruction_seq


def mpCalFuzzyHash(similarity_matrix, start_idx, method_list, instruction_seq):
    for kz_2 in range(start_idx + 1, len(method_list)):
        m1 = method_list[start_idx]
        m2 = method_list[kz_2]
        seq1 = "".join(instruction_seq[m1][1:]).replace('\n', '').replace("\t", " ")
        seq2 = "".join(instruction_seq[m2][1:]).replace('\n', '').replace("\t", " ")
        fuzzy_hash_value_1 = ppdeep.hash(seq1)
        fuzzy_hash_value_2 = ppdeep.hash(seq2)
        similarity = ppdeep.compare(fuzzy_hash_value_1, fuzzy_hash_value_2)
#         if 100> similarity > 90:
#             print(str(similarity) + "\t" + seq1 + "\t" + seq2)
        similarity_matrix[start_idx][kz_2] = similarity
        similarity_matrix[kz_2][start_idx] = similarity
    return similarity_matrix


def mpCalFuzzyHashOut(similarity_matrix, method_list, instruction_seq):
    for kz_1 in tqdm(range(0, len(method_list) - 1)):
        pool = ThreadPoolExecutor(max_workers=4)
        pool.submit(mpCalFuzzyHash, similarity_matrix, kz_1, method_list, instruction_seq)

def mpCalFuzzyHashOut_debug(similarity_matrix, method_list, instruction_seq):
    for kz_1 in tqdm(range(0, len(method_list) - 1)):
        mpCalFuzzyHash(similarity_matrix, kz_1, method_list, instruction_seq)



def fcg2adj(folderpath,sha256_name, nodes_list):

    filename = str(sha256_name)+"_testGraph.txt"

    fgraph = open(os.path.join(folderpath,filename), "r", encoding='utf-8')
    line = fgraph.readline()
    row_ind = []
    col_ind = []
    data = []
    while line:
        line = line.split("\n")[0]
        nodes = line.split(" ==> ")
        row_ind.append(nodes_list.index(nodes[0]))
        col_ind.append(nodes_list.index(nodes[1]))
        data.append(1)
        line = fgraph.readline()
    adj_matrix = coo_matrix((data, (row_ind, col_ind)), shape=[len(nodes_list), len(nodes_list)])
    return adj_matrix.toarray()

if __name__ == '__main__':
    apk_path = "./adv_apk"
    smali_dir = "./smaliOutput/"
      
    sensitive_api_file = "../data/malscan/sensitive_apis_soot_signature.txt"
    sensitive_api_list = []
    with open(sensitive_api_file,"r") as f:
        for line in f.readlines():
            sensitive_api_list.append(line.strip())
    
    exp_path = "./interpretation"
    save_path = "./de_obf_results"
    save_path_deletenode = save_path +"/delete_node"
    
    
    start = time.time()
    sha256_list = os.listdir(apk_path)
    
    for file in sha256_list:
        sha256 = file.split(".")[0]
        print(sha256)
        smali_output = os.path.join(smali_dir, sha256)
        
        exp_nodes = []
        flag = 0
        with open(os.path.join(exp_path,sha256+".txt"), "r") as f:
            for line in f.readlines():
                node = line.strip()
                for fram_name in framework:
                    if node.startswith(fram_name):
                        flag = 1
                        break
                if flag == 1:
                    continue
                exp_nodes.append(node)
                


        adv_apk = os.path.join(apk_path, sha256 + ".apk")

        # Analyze added nodes and added edges
        FLAG_analyze_Smali = True
        
        if os.path.exists(smali_output):
           FLAG_analyze_Smali = False 
        
        if FLAG_analyze_Smali:
            apk_tool_cmd = "apktool d %s -o %s" % (adv_apk, smali_output)

            p = subprocess.Popen(apk_tool_cmd, shell=True)

            time.sleep(13)
            p.kill()
            print("finish parse Smali!")
        
        # Find similar methods
        smali_folder = os.path.join(smali_output,"smali")
        instruction_seq = findSmalimethods(exp_nodes,smali_folder)

        method_list = list(instruction_seq)
        n = len(method_list)
        similarity_matrix = [[0 for k1 in range(n)] for k2 in range(n)]
        mpCalFuzzyHashOut_debug(similarity_matrix, method_list, instruction_seq)
        # pool = ThreadPoolExecutor(max_workers=16)
        # pool.submit(mpCalFuzzyHashOut, similarity_matrix, method_list, instruction_seq)
        
        if not os.path.exists(os.path.join(save_path_deletenode,sha256)):
            os.makedirs(os.path.join(save_path_deletenode,sha256))
            
        f = open(os.path.join(save_path_deletenode,sha256,"res.txt"),"w")
        for z1 in range(0, len(method_list) - 1):
            m1 = method_list[z1]
            tmp = similarity_matrix[z1]
            for z2 in range(z1 + 1, len(method_list)):
                m2 = method_list[z2]
                sim = tmp[z2]
                # if 60< sim <100:
                print("[Delete node] " + str(sim) + '\t' + m1 + ' -> ' + m2)
                f.write("[Delete node] " + str(sim) + '\t' + m1 + ' -> ' + m2 + '\n')
        f.close()
        
    end = time.time()
    print("Total time consumption: ", str(end-start))
  