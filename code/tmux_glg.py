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

    datasets = ["BAMultiShapes", "NCI1"] # ["BAMultiShapes", "MUTAG", "Mutagenicity", "NCI1"]
    archs = ["gin"] # ["gcn", "gat", "gin"]
    poolings = ["sum"] # ["sum", "mean", "max"]
    sizes = [1.0] # [0.05, 0.25, 0.5, 0.75]

    cpu = 30
    for dataset in datasets:
        for arch in archs:
            for pooling in poolings:
                if dataset == "NCI1":
                    seeds = [45, 1225, 1983]
                else:
                    seeds = [45, 357, 796]
                for seed in seeds:
                    for size in sizes:
                        #! passed -t
                        command=f"taskset -c {cpu} python glg_{dataset}.py -t -d cpu -s {seed} -r 0 --size {size} -e PGExplainer -a {arch} -p {pooling}"
                        log=f"../logs/acc_pool/{dataset}_PGExplainer_size{size}_seed{seed}_run_{arch}_{pooling}.log"
                        session_name = f"glg_{dataset}_{arch}_{pooling}_{seed}_{str(size).replace('.', '_')}"

                        # command = f"{command} &> {log}"
                        sessions[session_name] = ["conda activate glg", command]
                        cpu = (cpu + 1) % 95
    manage_tmux_sessions(sessions)
