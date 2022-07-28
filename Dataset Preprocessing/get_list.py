import os

path_labels ="/home/ashimag/wii_data_species_2022/labels/wii_all_labels/"
#we shall store all the file names in this list
path_images= "/home/ashimag/wii_data_species_2022/images/wii_data_aite_species_data-2022/"
file_labels = []
file_images=[]
path= "/home/ashimag/wii_data_species_2022/images/visualise_with_threshold/"
for root, dirs, files in os.walk(path_images):
	for file in files:

        #append the file name to the list
		file_images.append(os.path.join(root, file.split('.')[0]+'.jpg'))
        # file_images.append(os.path.join(root,file))
        
    # print(file_images)

print(file_images[0])   
print(len(file_images))    

for root, dirs, files in os.walk(path_labels):
	for file in files:
        #append the file name to the list
		file_labels.append(os.path.join(path_images, root.split('/')[6], file.split('.')[0]+'.jpg'))
        # file_labels.append(os.path.join(root,file.split('.')[0], '.jpg'))


print(file_labels[0])
print(len(file_labels))
no_bbox= list(set(file_images)-set(file_labels))
print(len(no_bbox))

with open('/home/ashimag/wii_data_species_2022/labels/no_bbox_imgs.txt', 'w') as fp:
        fp.write('\n'.join(no_bbox))

###########read all subfolder names################
classes= ['mani_cras-Manis crassicaudata', 'maca_munz-Macaca munzala', 'maca_radi-Macaca radiata', 'athe_macr', 'vulp_beng', 'lept_java-Leptoptilos javanicus',
 'trac_pile-Trachypithecus pileatus', 'hyst_brac-Hystrix brachyura', 'nilg_hylo-Nilgiritragus hylocrius', 'prio_vive-Prionailurus viverrinus',
  'neof_nebu-Neofelis nebulosa', 'melu_ursi', 'vehi_vehi', 'hyae_hyae-Hyaena hyaena', 'maca_mula-Macaca mulatta', 'fran_pond-Francolinus pondicerianus',
   'munt_munt-Muntiacus muntjak', 'feli_sylv-Felis sylvestris', 'maca_sile-Macaca silenus', 'vive_zibe-Viverra zibetha', 'rusa_unic-Rusa unicolor',
    'lepu_nigr-Lepus nigricollis', 'vive_indi-Viverricula indica', 'pavo_cris', 'anti_cerv', 'gall_lunu-Galloperdix lunulata', 'cato_temm-Catopuma temminckii',
     'sus__scro-Sus scrofa', 'cani_aure-Canis aureus', 'para_herm-Paradoxurus hermaphroditus', 'axis_axis', 'catt_kill', 'goat_sheep', 'vara_beng-Varanus bengalensis',
      'para-jerd-Paradoxurus jerdoni', 'mart_gwat-Martes gwatkinsii', 'homo_sapi', 'semn_john+Semnopithecus johnii', 'herp_edwa-Herpestes edwardsii', 'bos__fron',
       'herp_vitt-Herpestes vitticollis', 'arct_coll', 'dome_cats-Domestic cat', 'bos__indi', 'mell_cape-Mellivora capensis', 'ursu_thib-Ursus thibetanus',
        'semn_ente-Semnopithecus entellus', 'prio_rubi-Prionailurus rubiginosus', 'dome_dogs-Domestic dog', 'cani_lupu-Canis lupus', 'gall_sonn-Gallus sonneratii',
         'gaze_benn-Gazella bennettii', 'bose_trag-Boselaphus tragocamelus', 'budo_taxi-Budorcas taxicolor', 'bos__gaur', 'catt_catt-Cattle', 'blan_blan',
          'cuon_alpi-Cuon alpinus', 'capr_thar-Capricornis thar', 'equu_caba-Equus caballus', 'herp_fusc-Herpestes fuscus', 'trac_john-Trachypithecus johnii',
           'vara_salv-Varanus salvator', 'gall_gall-Gallus gallus', 'naem_gora-Naemorhedus goral', 'herp_urva-Herpestes urva', 'hyst_indi-Hystrix indica',
            'herp_smit-Herpestes smithii', 'bird_bird', 'tetr_quad-Tetracerus quadricornis', 'feli_chau-Felis chaus', 'maca_arct-Macaca arctoides',
             'lutr_pers-Lutrogale perspicillata', 'mosc_indi-Moschiola indica', 'pant_tigr', 'pant_pard-Panthera pardus', 'mart_flav-Martes flavigula',
              'pagu_larv-Paguma larvata-Masked Palm Civet', 'prio_beng-Prionailurus bengalensis', 'gall_spad-Galloperdix spadicea', 'elep_maxi-Elephas maximus',
               'axis_porc']
# subfolders=[]
# for root, dirs, files in os.walk(path_images):
#     subfolders.append(dirs)

# rm_dir = list(set(subfolders[0])-set(classes))
# print(rm_dir)
# print(len(rm_dir))


path= "/home/ashimag/wii_data_species_2022/images/visualise_with_threshold/"
subfolders=[]
for root, dirs, files in os.walk(path):
    subfolders.append(dirs)
# print(len(subfolders[0]))