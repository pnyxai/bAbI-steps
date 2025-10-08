**Create configs with**

* default

```bash
cd babisteps/lm_eval/
python3 _generate_configs.py --base_yaml_path ./default/_default_template_yaml --save_prefix_path "./default/babisteps"
```


* chat-cot

```bash
cd babisteps/lm_eval/
python3 _generate_configs.py --base_yaml_path ./chat-cot/_default_template_yaml --save_prefix_path "./chat-cot/babisteps" --task_prefix "chat-cot"
```