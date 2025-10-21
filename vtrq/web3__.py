import os

from web3 import Web3


w3 = Web3(Web3.HTTPProvider('http://192.168.174.129:8545'))

# 检查是否成功连接到节点
if  w3.is_connected():
    print("连接到以太坊私有链")

# abi = [
#     {
#       "inputs": [],
#       "name": "storedHash1",
#       "outputs": [
#         {
#           "internalType": "bytes32",
#           "name": "",
#           "type": "bytes32"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     },
#     {
#       "inputs": [],
#       "name": "storedHash2",
#       "outputs": [
#         {
#           "internalType": "bytes32",
#           "name": "",
#           "type": "bytes32"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "bytes32",
#           "name": "_hash1",
#           "type": "bytes32"
#         },
#         {
#           "internalType": "bytes32",
#           "name": "_hash2",
#           "type": "bytes32"
#         }
#       ],
#       "name": "setHashes",
#       "outputs": [],
#       "stateMutability": "nonpayable",
#       "type": "function"
#     },
#     {
#       "inputs": [],
#       "name": "getHashes",
#       "outputs": [
#         {
#           "internalType": "bytes32",
#           "name": "",
#           "type": "bytes32"
#         },
#         {
#           "internalType": "bytes32",
#           "name": "",
#           "type": "bytes32"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     }
#   ]

abi = [
    {
      "inputs": [],
      "name": "storedHash",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": True
    },
    {
      "inputs": [
        {
          "internalType": "bytes32",
          "name": "_hash",
          "type": "bytes32"
        }
      ],
      "name": "setHash",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getHash",
      "outputs": [
        {
          "internalType": "bytes32",
          "name": "",
          "type": "bytes32"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": True
    }
  ]
# contract_address = "0xF57402a7F3b0187747E79a051bE52FaD6198C84d"
contract_address = "0xA3d6baa25af0EE8e960A6eb287e05b146C022FE1"
contract = w3.eth.contract(address=contract_address, abi=abi)

import time
def generate_random_bytes32():
    """生成随机的32字节哈希值（bytes32格式）"""
    # 生成32字节随机数据
    random_bytes = os.urandom(32)

    # 转换为带0x前缀的十六进制字符串
    hex_str = Web3.to_hex(random_bytes)

    # 确保长度为66字符（0x + 64个字符）
    assert len(hex_str) == 66, f"生成的哈希长度不正确: {len(hex_str)}"

    return hex_str
import threading
def write_data_to_contract():
    # 发送交易的账户（确保有足够ETH支付gas）


    start_time = time.time()
    # 要写入的哈希值（示例）
    hash1 = generate_random_bytes32()
    # hash2 = generate_random_bytes32()
    bytes32_hash1 = Web3.to_bytes(hexstr=hash1)
    # bytes32_hash2 = Web3.to_bytes(hexstr=hash2)
    # 构建交易
    tx = contract.functions.setHash(bytes32_hash1).build_transaction({
        'from': w3.eth.accounts[0],
        'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0]),
        'gas': 2000000,
        'gasPrice': w3.eth.gas_price
    })
    # 发送交易
    tx_hash = w3.eth.send_transaction(tx)
    print(f"交易已发送，哈希: {tx_hash.hex()}")
    # 等待交易确认
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Deployment Gas: {tx_receipt.gasUsed}")
    print(f"交易确认，区块号: {tx_receipt.blockNumber}")
    end_time = time.time()
    print("数据上传时间",end_time - start_time)
    return end_time , tx_receipt.blockNumber

    # start_time2 = time.time()
    # nodeRpc = "http://127.0.0.1:8545"
    # while True:
    #     # 创建Web3实例连接到目标节点
    #     web3 = Web3(Web3.HTTPProvider(nodeRpc))
    #
    #     # 获取指定区块号的区块信息
    #     block = web3.eth.get_block(tx_receipt.blockNumber)
    #     if block:
    #         break
    #
    # end_time2 = time.time()
    # print("同步时间",end_time2 - start_time2)



    # 验证写入结果
    # current_hashes = contract.functions.getHashes().call()
    # hash1_hex = Web3.to_hex(current_hashes[0])
    # hash2_hex = Web3.to_hex(current_hashes[1])
    #
    # print(f"哈希1（十六进制）: {hash1_hex}")
    # print(f"哈希2（十六进制）: {hash2_hex}")
from web3.exceptions import BlockNotFound
def check_node_sync(nodeRpc, targetBlock, end_time, sync_times, lock):
    start_time2 = end_time
    # nodeRpc = "http://127.0.0.1:8545"
    while True:
        # 创建Web3实例连接到目标节点
        web3 = Web3(Web3.HTTPProvider(nodeRpc))

        # 获取指定区块号的区块信息
        try:
            block = web3.eth.get_block(targetBlock)
            if block:
                break
        except BlockNotFound:
            print(f"区块 {targetBlock} 尚未同步，等待中...")


    end_time2 = time.time()
    with lock:
        sync_times[nodeRpc] = {
            'finish_time': end_time2 - start_time2,
            }  # 相对于检查开始的时间

def check_multiple_nodes_sync(target_nodes, target_block, end_time):
    sync_times = {}
    lock = threading.Lock()
    threads = []

    # 创建并启动所有线程
    for nodeRpc in target_nodes:
        thread = threading.Thread(
            target=check_node_sync,
            args=(nodeRpc, target_block, end_time, sync_times, lock)
        )
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    return sync_times


if __name__ == "__main__":
#     # 写入数据
    end_time, target_block = write_data_to_contract()
    target_nodes = [
        "http://192.168.174.129:8545",
        "http://192.168.174.129:8546",
        "http://192.168.174.129:8547",
        "http://192.168.174.129:8548",
        "http://192.168.174.128:8545",
        "http://192.168.174.128:8546",
        "http://192.168.174.128:8547",
        "http://192.168.174.128:8548",

    ]
    results = check_multiple_nodes_sync(target_nodes, target_block, end_time)
    print(results)




#
# abi = [
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "",
#           "type": "uint256"
#         }
#       ],
#       "name": "locationHashes",
#       "outputs": [
#         {
#           "internalType": "int256",
#           "name": "minLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "minLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "string",
#           "name": "hash",
#           "type": "string"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "int256",
#           "name": "_minLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "_maxLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "_minLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "_maxLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "string",
#           "name": "_hash",
#           "type": "string"
#         }
#       ],
#       "name": "addLocationHash",
#       "outputs": [],
#       "stateMutability": "nonpayable",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "_index",
#           "type": "uint256"
#         }
#       ],
#       "name": "deleteLocationHash",
#       "outputs": [],
#       "stateMutability": "nonpayable",
#       "type": "function"
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "uint256",
#           "name": "_index",
#           "type": "uint256"
#         }
#       ],
#       "name": "getLocationHash",
#       "outputs": [
#         {
#           "internalType": "int256",
#           "name": "minLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "minLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "string",
#           "name": "hash",
#           "type": "string"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     },
#     {
#       "inputs": [],
#       "name": "getLocationHashCount",
#       "outputs": [
#         {
#           "internalType": "uint256",
#           "name": "",
#           "type": "uint256"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     },
#     {
#       "inputs": [
#         {
#           "internalType": "int256",
#           "name": "minLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLatitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "minLongitude",
#           "type": "int256"
#         },
#         {
#           "internalType": "int256",
#           "name": "maxLongitude",
#           "type": "int256"
#         }
#       ],
#       "name": "findSingleHashByCoordinateRanges",
#       "outputs": [
#         {
#           "internalType": "string",
#           "name": "",
#           "type": "string"
#         }
#       ],
#       "stateMutability": "view",
#       "type": "function",
#       "constant": True
#     }
#   ]
#
# # 合约部署后的地址，需要替换为实际的合约地址
# contract_address = "0xdDCbc39C4bDAA6D7ed766d18A914eCFde76eBBde"
#
# # 创建合约实例，结合地址和 ABI 来调用合约函数
# contract = w3.eth.contract(address=contract_address, abi=abi)
#
#
# # 连接到以太坊节点，根据实际情况修改地址和端口
# w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8545'))
#
# # 检查是否成功连接到节点
# if not w3.is_connected():
#     print("无法连接到以太坊节点，请检查节点是否运行。")
#     exit()
#
# # 合约的 ABI（应用二进制接口）
# abi = [
#     {
#         "inputs": [
#             {
#                 "internalType": "int256",
#                 "name": "_minLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "_maxLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "_minLongitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "_maxLongitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "string",
#                 "name": "_hash",
#                 "type": "string"
#             }
#         ],
#         "name": "addLocationHash",
#         "outputs": [],
#         "stateMutability": "nonpayable",
#         "type": "function"
#     },
#     {
#         "inputs": [
#             {
#                 "internalType": "uint256",
#                 "name": "_index",
#                 "type": "uint256"
#             }
#         ],
#         "name": "getLocationHash",
#         "outputs": [
#             {
#                 "internalType": "int256",
#                 "name": "minLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "maxLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "minLongitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "maxLongitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "string",
#                 "name": "hash",
#                 "type": "string"
#             }
#         ],
#         "stateMutability": "view",
#         "type": "function"
#     },
#     {
#         "inputs": [],
#         "name": "getLocationHashCount",
#         "outputs": [
#             {
#                 "internalType": "uint256",
#                 "name": "",
#                 "type": "uint256"
#             }
#         ],
#         "stateMutability": "view",
#         "type": "function"
#     },
#     {
#         "inputs": [
#             {
#                 "internalType": "int256",
#                 "name": "minLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "maxLatitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "minLongitude",
#                 "type": "int256"
#             },
#             {
#                 "internalType": "int256",
#                 "name": "maxLongitude",
#                 "type": "int256"
#             }
#         ],
#         "name": "findSingleHashByCoordinateRanges",
#         "outputs": [
#             {
#                 "internalType": "string",
#                 "name": "",
#                 "type": "string"
#             }
#         ],
#         "stateMutability": "view",
#         "type": "function"
#     }
# ]
#
# # 合约部署后的地址，需要替换为实际的合约地址
# contract_address = "0xdDCbc39C4bDAA6D7ed766d18A914eCFde76eBBde"
#
# # 创建合约实例
# contract = w3.eth.contract(address=contract_address, abi=abi)
#
#
# # 向合约写入数据
# def write_data_to_contract():
#     try:
#         with open('data.txt', 'r') as file:
#             lines = file.readlines()
#             for line in lines:
#                 # 假设每行数据格式为：minLatitude,maxLatitude,minLongitude,maxLongitude,hash
#                 data = line.strip().split(',')
#                 min_latitude = int(float(data[0]) * 1e6)
#                 max_latitude = int(float(data[1]) * 1e6)
#                 min_longitude = int(float(data[2]) * 1e6)
#                 max_longitude = int(float(data[3]) * 1e6)
#                 hash_str = data[4]
#
#                 # 调用合约的 addLocationHash 函数
#                 txn = contract.functions.addLocationHash(
#                     min_latitude,
#                     max_latitude,
#                     min_longitude,
#                     max_longitude,
#                     hash_str
#                 ).build_transaction({
#                     'from': w3.eth.accounts[0],
#                     'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0]),
#                     'gas': 2000000,
#                     'gasPrice': w3.eth.gas_price
#                 })
#                 # 发送交易
#                 txn_hash = w3.eth.send_transaction(txn)
#                 txn_receipt = w3.eth.wait_for_transaction_receipt(txn_hash)
#                 print(f"Transaction hash: {txn_receipt.transactionHash.hex()}")
#     except FileNotFoundError:
#         print("未找到 TXT 文件，请检查文件路径。")
#     except Exception as e:
#         print(f"发生错误: {e}")
#
#
# # 从合约查询数据
# # 根据输入的经纬度查询数据
# def query_data_from_contract():
#     try:
#         # 获取用户输入的经纬度数据
#         min_latitude_input = float(32.5678)
#         max_latitude_input = float(33.0087)
#         min_longitude_input = float(122.0456)
#         max_longitude_input = float(123.00915)
#
#         # 将输入的经纬度转换为整数
#         min_latitude = int(min_latitude_input * 1e6)
#         max_latitude = int(max_latitude_input * 1e6)
#         min_longitude = int(min_longitude_input * 1e6)
#         max_longitude = int(max_longitude_input * 1e6)
#
#         # 调用合约的 findSingleHashByCoordinateRanges 函数
#         result = contract.functions.findSingleHashByCoordinateRanges(
#             min_latitude,
#             max_latitude,
#             min_longitude,
#             max_longitude
#         ).call()
#
#         if result:
#             print(f"找到匹配的记录，对应的字符串为: {result}")
#         else:
#             print("未找到匹配的记录。")
#
#     except Exception as e:
#         print(f"查询数据时发生错误: {e}")
#
#
