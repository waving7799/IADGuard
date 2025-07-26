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


if __name__ == '__main__':
    apk_path = "./adv_apk"
    exp_path = "./interpretation"

    sha256_list = os.listdir(apk_path)
    start = time.time()
    FLAG_extract_deobf = True

    for sha256_name in sha256_list:
        sha256 = sha256_name.split(".")[0]
        print(sha256)


        adv_apk = os.path.join(apk_path, sha256 + ".apk")
        save_path = "./de_obf_results"
        save_path_deledge = save_path +"/delete_edge"
        save_path_addedge = save_path +"/add_edge"
        save_path_addnode = save_path +"/add_node"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        
        # Analyze added nodes and added edges
        apk = adv_apk
        start = time.time()
        if FLAG_extract_deobf:
            android_jars = "./android_jars"
            
            fcg_cmd = "java -Xmx50g -jar ./AddNode.jar -apkName %s -apkBasePath %s -outputBasePath %s " \
                      "-androidPath %s -explainNodesFolder %s  -debugMode 0 -isJimpleOutput 0" % (sha256+".apk", apk_path, save_path_addnode, android_jars, exp_path)
            os.system(fcg_cmd)
            
            fcg_cmd = "java -Xmx50g -jar ./AddEdge.jar -apkName %s -apkBasePath %s -outputBasePath %s " \
                      "-androidPath %s -explainNodesFolder %s  -debugMode 1 -isJimpleOutput 0" % (sha256+".apk", apk_path, save_path_addedge, android_jars, exp_path)
                      
            os.system(fcg_cmd)
            
            fcg_cmd = "java -Xmx50g -jar ./DeleteEdge.jar -apkName %s -apkBasePath %s -outputBasePath %s " \
                      "-androidPath %s -explainNodesFolder %s  -debugMode 1 -isJimpleOutput 0" % (sha256+".apk", apk_path, save_path_deledge, android_jars, exp_path)
                      
            os.system(fcg_cmd)
            

        
    final_end = time.time()
    print("Total time consumption: ", str(final_end-start))

