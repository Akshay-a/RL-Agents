This Notebook tries to show demo working of hot swapping:

Suppose we have a AI App which has 10 diff clients
who want to have diff domain driven behaviours -
Traditional way if we have seperate LLM for each client it would incur lot of cost to serve each client, instead we rely on Lora Hot Swapping feature 
In Above case:
We load the single, shared base model into the GPU just once.

We load the small LoRA adapter files for each of our ten customers into memory.

When a request comes in from "Customer A", we tell the base model: "Hey, for this request, please use Customer A's adapter."

When the next request comes from "Customer B", we instantly tell the model: "Okay, now use Customer B's adapter."

This is "hot swapping". We're dynamically activating different adapters for each inference request without ever reloading the massive base model. This saves an enormous amount of memory and makes serving many custom models incredibly fast and cheap.