import os
import re

def parse_cmd_args_log(filepath):
    params = {}
    with open(filepath, 'r') as file:
        for line in file:
            print(f"Analizzando linea di cmd-args.log: {line.strip()}")
            # Modifica la regex se i parametri non seguono il formato `--param=value`
            match = re.match(r"--(\w+)=([\w\.]+)", line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2)
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
                params[param_name] = param_value
    return params

def find_best_params_in_logs(directory):
    best_loss = float('inf')
    best_params = {}
    best_run = None

    for root, dirs, files in os.walk(directory):
        for dirname in dirs:
            cmd_args_path = os.path.join(root, dirname, "cmd-args.log")
            lightning_logs_path = os.path.join(root, dirname, "lightning_logs.txt")

            # Estrai iperparametri da cmd-args.log
            if os.path.exists(cmd_args_path):
                params = parse_cmd_args_log(cmd_args_path)
            else:
                print(f"{cmd_args_path} non trovato, si passa alla directory successiva.")
                continue

            # Estrai la metrica (ad esempio, val_loss) da lightning_logs.txt
            val_loss = None
            if os.path.exists(lightning_logs_path):
                with open(lightning_logs_path, 'r') as log_file:
                    for line in log_file:
                        print(f"Analizzando linea di lightning_logs.txt: {line.strip()}")
                        match = re.search(r"val_loss\s*=\s*([\d\.]+)", line)
                        if match:
                            val_loss = float(match.group(1))
                            break

            # Verifica se è stato trovato un val_loss valido
            if val_loss is not None:
                print(f"Val_loss trovato in {dirname}: {val_loss}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
                    best_run = dirname

    if best_run:
        print(f"Miglior valore di loss trovato in '{best_run}': {best_loss}")
        print("I migliori iperparametri trovati sono:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    else:
        print("Nessun risultato valido trovato nei file di log nella directory specificata.")

if __name__ == '__main__':
    directory_path = "../experiments/asr_experiment/2024-11-13_18-06-26"
    find_best_params_in_logs(directory_path)
