import json
from graph import create_graph
from dotenv import load_dotenv

load_dotenv()

def main():
    graph = create_graph()
    user_input = "create prompt to write technical article for senior software engineers"
    model_name = "llama3-70b-8192"
    res = graph.invoke({"user_input": user_input, "model_name": model_name})
    output_data = {
        "user_input": user_input,
        "initial_prompt": res.get("initial_prompt"),
        "evaluations": res.get("evaluations"),
        "final_prompt": res.get("final_prompt"),
        "model_name": model_name
    }
    json_file_path = "output.json"
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = []

    data.append(output_data)

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print("Data has been appended to", json_file_path)

if __name__ == "__main__":
    main()
