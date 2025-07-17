# End-to-End Model Optimization: Knowledge Distillation and Quantization

This project demonstrates a complete, end-to-end workflow for optimizing a large NLP model for production. Starting with a fine-tuned BERT-base model, we use a two-stage optimization process to create a final model that is significantly smaller, faster, and cheaper to run, all while maintaining high performance.

The two main techniques used are:
1.  **Knowledge Distillation:** A smaller "student" model (DistilBERT) is trained to mimic the behavior of a larger "teacher" model (BERT-base), effectively transferring its knowledge into a more compact architecture.
2.  **Post-Training Dynamic Quantization:** The distilled model is further optimized by converting its weights from 32-bit floating-point numbers to 8-bit integers, using the ONNX Runtime for deployment-ready performance.

The entire process, including data loading, custom trainer implementation, and evaluation, is documented in the accompanying Jupyter Notebook.

## Project Links
*   **Live Notebook:** [https://github.com/vyshnev/Model-Distillation/blob/main/knowledge-distillation%20and%20quantization/kd_and_quantization.ipynb]
*   **Fine-Tuned Student Model:** [`Vyshnev/distilbert-base-uncased-finetuned-clinc-student`](https://huggingface.co/Vyshnev/distilbert-base-uncased-finetuned-clinc-student)
*   **Quantized ONNX Student Model:** [`Vyshnev/distilbert-base-uncased-clinc-student-quantized-onnx`](https://huggingface.co/Vyshnev/distilbert-base-uncased-clinc-student-quantized-onnx)

## The Problem: Large Models in Production

Large Transformer models like BERT are powerful but come with significant drawbacks for production environments:
*   **High Latency:** They are slow to respond, which can lead to a poor user experience.
*   **High Computational Cost:** They require expensive hardware (like GPUs) to run efficiently, increasing operational costs.
*   **Large Memory Footprint:** Their size makes them difficult to deploy on resource-constrained devices like mobile phones or edge servers.

This project tackles these challenges directly by creating a highly efficient version of the model.

## The Solution: A Two-Stage Optimization Pipeline

The core of this project is a pipeline that systematically reduces the model's size and improves its speed.

### Stage 1: Knowledge Distillation
A custom `KnowledgeDistillationTrainer` was implemented by subclassing the standard Hugging Face `Trainer`. This new trainer uses a specialized loss function that combines the standard cross-entropy loss (learning from the true labels) with a distillation loss (learning from the teacher's probability distributions).

- **Teacher Model:** `transformersbook/bert-base-uncased-finetuned-clinc`
- **Student Model:** `distilbert-base-uncased`

### Stage 2: Post-Training Quantization
The resulting distilled student model was then exported to the **ONNX (Open Neural Network Exchange)** format, a standard for interoperable AI models. Using the `optimum` library, we applied post-training dynamic quantization, which converts the model's weights to the efficient INT8 data type. This step provides a significant speed boost for CPU-based inference.

## Final Results: A Smaller, Faster, High-Performance Model

The optimization pipeline was a definitive success. The final quantized model is a fraction of the size and multiple times faster than the original teacher model, with only a negligible drop in accuracy.

| Model | Parameters | On-Disk Size (MB) | Avg. Latency (CPU) | Validation Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Teacher (BERT-base)** | 110M | 439 MB | 67.65 ms | ~93%* |
| **Student (Distilled)** | 67M (-39%) | 256 MB (-42%) | 60.37 ms | **92.48%** |
| **Student (Distilled + Quantized)**| 67M (-39%) | **64 MB (-85%)** | **13.25 ms (5.1x speedup)** | **91.87%** (-0.61%) |

*\*Teacher model accuracy is based on its original reported performance.*

### Key Takeaways:
*   **Size:** The final model is **~85% smaller** than the original, making it far easier to store and deploy.
*   **Speed:** The final model is **over 5 times faster** on a CPU, dramatically reducing inference costs and improving responsiveness.
*   **Performance:** The entire optimization process resulted in an accuracy drop of less than 1%, demonstrating a highly effective trade-off.

## How to Run This Project

1.  Clone the repository:
    ```bash
    git clone hhttps://github.com/vyshnev/Model-Distillation.git
    cd Model-Distillation/knowledge-distillation and quantization
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open and run the Jupyter Notebook `knowledge_distillation_implementation_end_to_end.ipynb` in an environment like Google Colab or a local Jupyter server.

## Technologies Used
- Python
- PyTorch
- Hugging Face `transformers`, `datasets`, `evaluate`, and `optimum`
- ONNX & ONNX Runtime
- Jupyter Notebooks