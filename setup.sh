#Defines utility var
BLENDER_ARCHIVE=blender-4.5.8-linux-x64.tar.xz

#Install blender 4.5.8 LTS (possible upgrade: install latest 4.5 version)
curl -L -O "https://www.blender.org/download/lts/4-5/${BLENDER_ARCHIVE}" 

#unzip blender
tar xf ${BLENDER_ARCHIVE} && mv ${BLENDER_ARCHIVE} blender

# Start blender
./blender/blender -h