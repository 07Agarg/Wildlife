import shutil,os
path_txt = "/home/ashimag/wii_data_species_2022/labels/no_bbox_imgs.txt"
img_path="/home/ashimag/wii_data_species_2022/images/visualise_with_threshold/"
dest_path= "/home/ashimag/wii_data_species_2022/images/rm_files/"
# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
with open(path_txt) as f:
    contents = f.readlines()
file_images=[]
path= "/home/ashimag/wii_data_species_2022/images/visualise_with_threshold/"
for root, dirs, files in os.walk(path):
	for file in files:

        #append the file name to the list
		file_images.append(os.path.join(root, file.split('.')[0]+'.jpg'))
        # file_images.append(os.path.join(root,file))
        
    # print(file_images)
add="anno_~home~ashimag~wii_data_species_2022~images~wii_data_aite_species_data-2022~blan_blan~"
# print(file_images[0])   
print(len(file_images))
for item in contents:
    dir_= item.split('/')[6]
    # print(dir_)
    img_name= item.split('/')[-1]
    # print(img_name)
    current_path= os.path.join(img_path,dir_,add+img_name)
    print(current_path)
    final_path= os.path.join(dest_path,dir_,img_name)
    print(final_path)
    # break
    # print(os.path.join(dest_path,dir_))
    if not os.path.exists(os.path.join(dest_path,dir_)):
        os.mkdir(os.path.join(dest_path,dir_))

    # shutil.move(current_path, final_path)
    # break

print('done')

file_images=[]
path= "/home/ashimag/wii_data_species_2022/images/visualise_with_threshold/"
for root, dirs, files in os.walk(path):
	for file in files:

        #append the file name to the list
		file_images.append(os.path.join(root, file.split('.')[0]+'.jpg'))
        # file_images.append(os.path.join(root,file))
        
    # print(file_images)

# print(file_images[0])   
print(len(file_images))