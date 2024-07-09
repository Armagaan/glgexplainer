import subprocess

def create_tmux_session(session_name):
    """Create a new tmux session."""
    try:
        subprocess.run(["tmux", "kill-session", "-t", session_name])
    except:
        print("No session to kill, creating ", session_name)
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name])

def run_command_in_tmux(session_name, command):
    """Run a command in an existing tmux session."""
    subprocess.run(["tmux", "send-keys", "-t", session_name, command, "C-m"])

def manage_tmux_sessions(sessions):
    """Create multiple tmux sessions and run commands in them."""
    # subprocess.run(["conda", "activate", "shap"])
    for session_name, commands in sessions.items():
        create_tmux_session(session_name)
        for command in commands:
            run_command_in_tmux(session_name, command)

if __name__ == "__main__":
    # Define the sessions and their respective commands
    sessions = {}

    datasets = ["MUTAG", "Mutagenicity"] # ["MUTAG", "Mutagenicity"]
    archs = ["gat"] # ["gcn", "gat", "gin"]
    poolings = ["sum"] # ["mean", "max"]
    sizes = [1.0] # [0.05, 0.25, 0.5, 0.75, 1.0]

    total_cpus = 96
    start_cpu = 40
    in_hand_cpus = total_cpus - start_cpu
    cpu = start_cpu
    counter = 0
    for dataset in datasets:
        for arch in archs:
            for pooling in poolings:
                if dataset == "NCI1":
                    seeds = [45, 1225, 1983]
                else:
                    seeds = [45, 357, 796]
                for seed in seeds:
                    for size in sizes:
                        command=f"taskset -c {cpu} python glg_iso.py -e PGExplainer --split test -r 0 -d {dataset} -a {arch} -p {pooling} -s {seed} --size {size}"
                        log = f"../logs/glg_iso/pgexplainer_{dataset}_{arch}_{pooling}_{size}_{seed}"
                        
                        session_name = f"glg_iso_{dataset}_{arch}_{pooling}_{int(100 * size)}_{seed}"
                        command = f"{command} > {log}_1.log 2> {log}_2.log"

                        if arch == "gin":
                            sessions[session_name] = ["conda activate pyg", command]
                        else:
                            sessions[session_name] = ["conda activate glg", command]
                        cpu = start_cpu + ((counter + 1) % in_hand_cpus)
                        counter += 1
    manage_tmux_sessions(sessions)
