# Post-Lesson Reflection Report: AAE 5303 Robust Control Technology

**Student Name:** YAN SHENTAO
**Student ID:** 25132971 G 
**Group Number:** ALateFix
**Date:** 25/04/2026

---

## Section 1: AI Usage Experience

In this project, I used **Cursor** as more than just a code editor; it acted as a **remote DevOps assistant** and **performance tuner**. 

*   **Specific Tasks:** 
    *   **Training Management:** I used AI to execute and monitor long-running training jobs for the `amtown_pipeline.py` script. 
    *   **Dynamic Refactoring:** When my initial training failed, I asked Cursor to refactor the training loop to support **Gradient Accumulation** and **Automatic Mixed Precision (AMP)** to handle memory constraints.
    *   **Live Monitoring:** I used the AI to "poll"  the output directories and terminal logs, providing me with structured snapshots of `train_loss`, `val_dice`, and `val_mIoU` while I was away from my computer.
*   **Frequency:** I used it **daily**, especially during the 65-epoch training runs, treating the AI as a bridge between the remote server and my local interface.
*   **Useful Features:** The **Chat-to-Terminal** integration was vital. It allowed me to run commands like `ps -ef | rg "python"` to verify process health and `ls -lah` to check if checkpoints (`.pth` files) were being saved correctly.

---

## Section 2: Understanding AI Limitations

A major limitation surfaced during the initial setup of the **U-Net training pipeline**.

*   **The Hallucination/Wrong Assumption:** Initially, I attempted to run the training with a `--batch-size 128`. While the AI generated the command correctly, it did not proactively warn me about the physical memory limits of the GPU for a U-Net model on the AMTown dataset until the process crashed with an "Out of Memory" (OOM) error.
*   **The Monitoring Delay:** During the polling process, the AI initially reported that the output directory was empty because it was looking at a stale terminal session. I had to manually intervene and point it to the "active window" to get the real-time progress.
*   **Detection & Fix:** I detected the issue through the crash logs. To fix it, I instructed the AI to implement a more robust strategy: reducing the physical batch size to **8** while using `accum-steps 16` to maintain an effective batch size of 128. I also forced the use of `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent memory fragmentation.

---

## Section 3: Engineering Validation

I validated the AI’s suggested "Anti-Explosion" configuration through rigorous monitoring of performance metrics:

1.  **Metric Tracking:** I didn't just check if the code ran; I tracked the **val_mIoU** and **val_dice** across epochs. I saw the `val_mIoU` climb from **0.1144** (Epoch 1) to **0.3629** (Epoch 6), which confirmed that the gradient accumulation logic was mathematically equivalent to the original large-batch requirement.
2.  **Checkpoint Verification:** I ensured that the `checkpoint_every 5` argument was respected. I manually verified the existence of `checkpoint_epoch5.pth` and `best_model.pth` in the `/home/daism/unet_minimal/amtown_outputs_65ep_fast` directory.
3.  **Process Health Checks:** I used the AI to monitor the system PID to ensure the training didn't silently hang, ensuring that the "30 s timeout" in the terminal didn't mean the background process had failed.

---

## Section 4: Problem-Solving Process

**Challenge: Resolving "Memory Explosion" (OOM) in U-Net Training.**

*   **The Issue:** My U-Net training on the AMTown dataset was consistently failing due to the high resolution and large batch size requirements, leading to "CUDA Out of Memory" errors.
*   **My Approach:** Instead of simply lowering the batch size (which would hurt model convergence), I decided to implement **Gradient Accumulation**.
*   **AI's Role:** I asked Cursor to "Directly modify and rerun" the script. The AI assisted by:
    1.  Adding `--accum-steps` to the argument parser.
    2.  Modifying the `loss.backward()` and `optimizer.step()` logic to only update weights every $N$ steps.
    3.  Setting environment variables to optimize CUDA memory allocation.
*   **Reasoning:** My reasoning was that stability is more important than raw speed. By switching to a `batch-size 8` and `accum-steps 16` configuration, I successfully bypassed the hardware limit while achieving the same optimization trajectory as a 128-batch run.

---

## Section 5: Learning Growth

*   **Technical Skills:** I gained deep practical experience in **Deep Learning Ops**. I learned how to manage long-running background tasks on Linux and how to troubleshoot GPU memory fragmentation.
*   **Confidence:** At the start, an "OOM" error would have stopped my progress for hours. Now, I have a "toolkit" of strategies (AMP, Gradient Accumulation, `expandable_segments`) to handle large models on limited hardware.
*   **System Understanding:** I now understand the difference between a "terminal session timeout" and a "process failure." I can confidently use polling and logging to monitor models, allowing me to be productive even when I am not physically in front of my computer.

---

## Section 6: Critical Reflection

AI played the role of a **"Co-Pilot and Sentinel"** in my project. 

*   **Pros:** It allowed me to refactor my code for complex memory management in seconds. It also provided "as-a-service" monitoring, giving me the peace of mind to let the model train overnight.
*   **Cons:** The AI can be "blind" to the real-time state if not carefully directed. I learned that I must provide the AI with the **"Active Window"** context to get accurate information.
*   **Lesson Learned:** In the future, I will always start with a "smoke test" (low batch size) before scaling up, even if the AI suggests a higher value. I also learned that visualizing the loss curve (like the generated `training_curves.png`) is the only way to truly verify if the training is "healthy" beyond just "running."

---

## Section 7: Evidence

### 7.1 Successful Training Command (Anti-Explosion Config)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /home/unet_minimal/amtown_pipeline.py \
  --mode train \
  --epochs 100 \
  --batch-size 8 \
  --accum-steps 16 \
  --scale 0.25 \
  --lr 5e-5 \
  --output-dir /home/unet_minimal/amtown_outputs_65ep_fast
```

### 7.2 Log Snapshot (Validation of Growth)
The following log captured by the AI proves the model was learning and the memory was stable:
*   **Epoch 1:** val_dice=0.6361, val_mIoU=0.1144
*   **Epoch 5:** val_dice=0.8841, val_mIoU=0.3201
*   **Epoch 6:** val_dice=0.9166, val_mIoU=0.3629 (Best Model Updated)

### 7.3 Prompt Example for Monitoring
**Prompt:** "帮我检测最新进度，持续阅读最新的活跃窗口，报当前 epoch 百分比 + 最新 best mIoU。"
**AI Response:** "当前在 Epoch 7（约 69%），最新指标 Epoch 6: val_mIoU=0.3629, best_model 已更新。"

---

**End of Report**
