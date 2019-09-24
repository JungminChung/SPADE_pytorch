import os 
import glob
from shutil import copyfile

def folder_setting():
    pwd_dir = os.getcwd()
    results_dir = os.path.join(pwd_dir,'results')
    output_folders = os.listdir(results_dir)
    inner_folders = ['ckpt', 'images', 'source', 'summary']

    if len(output_folders) == 0 : 
        folder_name = 'output_0001'
        os.mkdir(os.path.join(results_dir, folder_name))
        output_folder = os.path.join(results_dir, folder_name)
        for f in inner_folders : 
            os.mkdir(os.path.join(output_folder, f))
        
        ckpt_folder = os.path.join(output_folder, 'ckpt')
        image_folder = os.path.join(output_folder, 'images')
        source_folder = os.path.join(output_folder, 'source')
        summary_folder = os.path.join(output_folder, 'summary')

        py_files = glob.glob('*.py')
        for py_file in py_files : 
            copyfile(os.path.join(pwd_dir, py_file), os.path.join(source_folder, py_file))

    else : 
        num = [] 
        for x in output_folders :
            num.append(int(x.split('_')[1]))

        folder_name = 'output_' + f'{str(max(num)+1).zfill(4)}'
        os.mkdir(os.path.join(results_dir, folder_name))
        output_folder = os.path.join(results_dir, folder_name)
        for f in inner_folders : 
            os.mkdir(os.path.join(output_folder, f))
        
        ckpt_folder = os.path.join(output_folder, 'ckpt')
        image_folder = os.path.join(output_folder, 'images')
        source_folder = os.path.join(output_folder, 'source')
        summary_folder = os.path.join(output_folder, 'summary')

        py_files = glob.glob('*.py')
        for py_file in py_files : 
            copyfile(os.path.join(pwd_dir, py_file), os.path.join(source_folder, py_file))
            
        
        return ckpt_folder, image_folder, source_folder, summary_folder