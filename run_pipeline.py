import os
import sys
import time
import argparse

from src.dataset import get_eurosat_datasets, generate_sample_visualizations
from train_teacher import train_teacher
from train_student import train_student
from quantize_student import run_quantization
from evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="End-to-End Edge Satellite Scene Classification Pipeline")
    parser.add_argument('--skip-teacher', action='store_true', help="Skip teacher training if checkpoint already exists")
    parser.add_argument('--skip-student', action='store_true', help="Skip student training if checkpoints already exist")
    parser.add_argument('--skip-quant', action='store_true', help="Skip quantization if INT8 checkpoint exists")
    parser.add_argument('--teacher-epochs', type=int, default=12, help="Number of teacher training epochs")
    parser.add_argument('--student-epochs', type=int, default=16, help="Number of student training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    args = parser.parse_args()

    pipeline_start = time.time()
    print("=" * 80)
    print("  ONBOARD SATELLITE SCENE & CLOUD CLASSIFICATION UNDER SENSOR NOISE")
    print("  End-to-End Autonomous Pipeline Execution")
    print("=" * 80)

    # ---------------------------------------------------------
    # Phase 1: Dynamic Noise-Aware Data Pipeline
    # ---------------------------------------------------------
    print("\n>>> PHASE 1: Dynamic Noise-Aware Data Pipeline & Dataset Prep")
    data_dir = './data'
    train_set, test_set = get_eurosat_datasets(root=data_dir, download=True)
    print(f"[Dataset] Official EuroSAT subset loaded: {len(train_set)} Train / {len(test_set)} Test patches (Batch size: {args.batch_size})")

    generate_sample_visualizations(
        root=data_dir,
        output_path='deliverables/sample_augmentations.png'
    )

    # ---------------------------------------------------------
    # Phase 2: Knowledge Distillation (Teacher-Student Architecture)
    # ---------------------------------------------------------
    print("\n>>> PHASE 2: Knowledge Distillation Architecture")
    teacher_ckpt = 'checkpoints/teacher_best.pth'
    if not (args.skip_teacher and os.path.exists(teacher_ckpt)):
        print("\n--- Training Teacher (ResNet-50) ---")
        train_teacher(
            epochs=args.teacher_epochs,
            batch_size=args.batch_size,
            checkpoint_dir='checkpoints'
        )
    else:
        print(f"[Teacher] Reusing existing checkpoint at {teacher_ckpt}")

    student_scratch_ckpt = 'checkpoints/student_scratch.pth'
    student_distill_ckpt = 'checkpoints/student_distilled_noise.pth'

    if not (args.skip_student and os.path.exists(student_scratch_ckpt) and os.path.exists(student_distill_ckpt)):
        print("\n--- Training Baseline Student from Scratch (No Distillation, Clean Data) ---")
        train_student(
            mode='scratch',
            epochs=args.student_epochs,
            batch_size=args.batch_size,
            checkpoint_dir='checkpoints'
        )

        print("\n--- Training Student with Knowledge Distillation + Dynamic Noise Pipeline ---")
        train_student(
            mode='distilled_noise',
            epochs=args.student_epochs,
            batch_size=args.batch_size,
            teacher_checkpoint=teacher_ckpt,
            checkpoint_dir='checkpoints'
        )
    else:
        print(f"[Student] Reusing existing checkpoints at {student_scratch_ckpt} and {student_distill_ckpt}")

    # ---------------------------------------------------------
    # Phase 3: Post-Training Quantization (PTQ)
    # ---------------------------------------------------------
    print("\n>>> PHASE 3: Post-Training Static Quantization (FP32 -> INT8)")
    int8_ckpt = 'checkpoints/student_int8.pth'
    if not (args.skip_quant and os.path.exists(int8_ckpt)):
        run_quantization(
            student_fp32_path=student_distill_ckpt,
            output_int8_path=int8_ckpt,
            num_calibration_batches=20
        )
    else:
        print(f"[Quantization] Reusing existing INT8 checkpoint at {int8_ckpt}")

    # ---------------------------------------------------------
    # Phase 4: Headless Benchmarking & Metric Generation
    # ---------------------------------------------------------
    print("\n>>> PHASE 4: Headless Benchmarking & Evaluation")
    run_evaluation(
        data_dir=data_dir,
        deliverables_dir='deliverables',
        checkpoints_dir='checkpoints'
    )

    total_time = time.time() - pipeline_start
    print(f"\n[Complete] Entire pipeline executed in {total_time / 60:.2f} minutes.")


if __name__ == '__main__':
    main()
