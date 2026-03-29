#Defines utility var
BLENDER_ARCHIVE=blender-4.5.8-linux-x64

#Install blender 4.5.8 LTS (possible upgrade: install latest 4.5 version)
curl -L -O "https://download.blender.org/release/Blender4.5/${BLENDER_ARCHIVE}.tar.xz" 

#unzip blender
tar xf ${BLENDER_ARCHIVE}.tar.xz && mv ${BLENDER_ARCHIVE} blender

# Start blender
./blender/blender -h