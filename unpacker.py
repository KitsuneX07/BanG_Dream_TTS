from cryptography.fernet import Fernet
import os, bz2


def decrypt_file(encrypted_data, key):
    f = Fernet(key)
    try:
        decrypted_data = f.decrypt(encrypted_data)
    except Exception as e:
        print(f"Error decrypting data: {e}")
        raise
    return decrypted_data


def decompress_file(data):
    try:
        decompressed_data = bz2.decompress(data)
    except Exception as e:
        print(f"Error decompressing data: {e}")
        raise
    return decompressed_data


def extract_package(key: bytes, package_name: str, target_dir: str):
    """
    解包并解密文件

    参数:
        key (bytes): 用于解密的密钥
        package_name (str): .package 包文件的路径
        target_dir (str): 解压后的文件目录路径

    返回:
        None
    """
    with open(package_name, 'rb') as package_file:
        while True:
            name_size_bytes = package_file.read(4)
            if len(name_size_bytes) < 4:
                print("End of file reached or incomplete read")
                break
            name_size = int.from_bytes(name_size_bytes, 'big')

            file_name_bytes = package_file.read(name_size)
            if len(file_name_bytes) < name_size:
                raise ValueError("Incomplete file name read")
            file_name = file_name_bytes.decode('utf-8')

            data_size_bytes = package_file.read(4)
            if len(data_size_bytes) < 4:
                raise ValueError("Incomplete data size read")
            data_size = int.from_bytes(data_size_bytes, 'big')

            encrypted_data = package_file.read(data_size)
            if len(encrypted_data) < data_size:
                raise ValueError("Incomplete encrypted data read")

            decrypted_data = decrypt_file(encrypted_data, key)

            decompressed_data = decompress_file(decrypted_data)

            relative_path = os.path.basename(file_name)
            output_file_path = os.path.join(target_dir, relative_path)

            os.makedirs(os.path.dirname(output_file_path), exist_ok = True)

            with open(output_file_path, 'wb') as output_file:
                output_file.write(decompressed_data)
            print(f"Successfully extracted and decrypted: {output_file_path}")


if __name__ == '__main__':
    example_key = b'Guess!'
    example_package = '户山香澄.package'
    example_target_dir = 'extracted'
    os.makedirs(example_target_dir, exist_ok = True)
    extract_package(example_key, example_package, example_target_dir)
