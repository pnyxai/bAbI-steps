Title: TODO

Abstract: TODO


Topics: TODO


## SimpleTracking
Descriptions: TODO

### ActosInLocation
```bash
python3 main.py \
    --task simpletracking \
    --locations bedroom,bathroom,kitchen \
    --actors Alice,Bob,Charlie \
    --q_stories 2 \
    --states_lenght 3 \
    --question where \
    --answer unknown \
    --verbosity info
```


### ActorWithObject
```bash
python3 main.py \
    --task simpletracking \
    --actors Alice,Bob,Charlie \
    --objects book,ball,pen \
    --q_stories 2 \
    --states_lenght 3 \
    --question what \
    --answer none \
    --verbosity debug
```


## ComplexTracking

### ObjectInLocationPolar

```bash
python3 main.py \
    --task complextracking \
    --locations bedroom,bathroom,kitchen \
    --actors Alice,Bob,Charlie \
    --objects book,ball,pen \
    --q_stories 3 \
    --states_lenght 20 \
    --question polar \
    --answer yes
```

### ObjectInLocationWhat

```bash
python3 main.py \
    --task complextracking \
    --locations x_0,x_1,x_2,x_3 \
    --actors y_0,y_1,y_2,y_3,y_4,y_5 \
    --objects z_0,z_1,z_2,z_3,z_4,z_5,z_6,z_7,z_8,z_9,z_10 \
    --q_stories 15 \
    --states_lenght 20 \
    --question where \
    --answer designated_location
```