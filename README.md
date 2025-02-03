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
    --verbose 1
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
    --verbose 1
```


## ComplexTracking

### ObjectInLocationPolar

```bash
python3 main.py \
    --task complextracking \
    --locations bedroom,bathroom,kitchen \
    --actors alice,bob,charlie \
    --objects book,ball,pen \
    --q_stories 3 \
    --states_lenght 5 \
    --question polar \
    --answer yes \
    --verbose 1
```
