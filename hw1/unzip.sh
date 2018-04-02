mkdir -p src
sudo cp /home/opt/ftp/hw1/* ./src
sudo chown xinglu:xinglu -R src
rm -rf build
mkdir -p build
cd src
for file in *.zip; do
    mkdir -p ../build/$file
    unzip -O cp936 -d ../build/$file $file
done

for file in *.rar; do
    mkdir -p ../build/$file
    unrar x $file ../build/$file
done
