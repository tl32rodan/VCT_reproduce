echo "$(pwd)" >> ./VCT.pth
SITE=$(python3 -c 'import site; print(site.USER_SITE)')
mv ./VCT.pth "$SITE/VCT.pth"

cd torchac/
python3 setup.py install --user
rm -rf build dist torchac_backend_cpu.egg-info
cd ../
