"""
SAM3 (Segment Anything Model 3) wrapper module.
Provides text prompt-based mask generation functionality.
"""

import os
import sys

# Setup triton mock BEFORE importing anything else (especially torch)
def _setup_triton_mock():
    """Create a mock triton module for Windows compatibility."""
    if sys.platform == "win32" and "triton" not in sys.modules:
        import types
        from importlib.machinery import ModuleSpec

        def create_mock_module(name, is_package=False):
            """Create a mock module with proper __spec__."""
            mod = types.ModuleType(name)
            mod.__spec__ = ModuleSpec(name, None, is_package=is_package)
            mod.__loader__ = None
            mod.__package__ = name if is_package else name.rsplit('.', 1)[0] if '.' in name else ''
            if is_package:
                mod.__path__ = []
            return mod

        # Create mock triton module (as package)
        triton = create_mock_module("triton", is_package=True)
        triton.jit = lambda *args, **kwargs: lambda f: f
        triton.autotune = lambda *args, **kwargs: lambda f: f
        triton.heuristics = lambda *args, **kwargs: lambda f: f
        triton.Config = lambda *args, **kwargs: None

        # Create mock triton.language module
        tl = create_mock_module("triton.language", is_package=False)
        tl.dtype = type("dtype", (), {})
        tl.constexpr = type("constexpr", (), {"__init__": lambda self, x: None})
        tl.int32 = "int32"
        tl.int64 = "int64"
        tl.float16 = "float16"
        tl.float32 = "float32"
        tl.bfloat16 = "bfloat16"

        for func_name in ['load', 'store', 'arange', 'zeros', 'full', 'maximum', 'minimum',
                          'exp', 'log', 'sqrt', 'abs', 'cdiv', 'where', 'sum', 'max', 'min',
                          'dot', 'trans', 'broadcast_to', 'reshape', 'expand_dims', 'view',
                          'program_id', 'num_programs', 'atomic_add', 'atomic_max', 'atomic_min',
                          'debug_barrier', 'multiple_of']:
            setattr(tl, func_name, lambda *args, **kwargs: None)

        triton.language = tl

        # Create triton.runtime mock
        triton_runtime = create_mock_module("triton.runtime", is_package=True)
        triton_runtime.driver = create_mock_module("triton.runtime.driver", is_package=False)
        triton.runtime = triton_runtime

        # Create triton.backends mock
        triton_backends = create_mock_module("triton.backends", is_package=True)
        triton_backends.compiler = create_mock_module("triton.backends.compiler", is_package=False)
        triton_backends.nvidia = create_mock_module("triton.backends.nvidia", is_package=True)
        triton_backends.nvidia.driver = create_mock_module("triton.backends.nvidia.driver", is_package=False)
        triton.backends = triton_backends

        # Create triton.compiler mock
        triton_compiler = create_mock_module("triton.compiler", is_package=True)
        triton_compiler.compiler = create_mock_module("triton.compiler.compiler", is_package=False)
        triton.compiler = triton_compiler

        # Register all modules
        sys.modules["triton"] = triton
        sys.modules["triton.language"] = tl
        sys.modules["triton.runtime"] = triton_runtime
        sys.modules["triton.runtime.driver"] = triton_runtime.driver
        sys.modules["triton.backends"] = triton_backends
        sys.modules["triton.backends.compiler"] = triton_backends.compiler
        sys.modules["triton.backends.nvidia"] = triton_backends.nvidia
        sys.modules["triton.backends.nvidia.driver"] = triton_backends.nvidia.driver
        sys.modules["triton.compiler"] = triton_compiler
        sys.modules["triton.compiler.compiler"] = triton_compiler.compiler

# Call immediately at module load time
_setup_triton_mock()

import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import warnings


def _setup_sam3_path():
    """Add SAM3 local path to sys.path if needed."""
    # Check for local SAM3 installation in common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, "sam3"),  # ./sam3/sam3
        os.path.join(script_dir, "sam3", "sam3"),  # nested
        os.path.join(os.path.dirname(script_dir), "sam3"),  # ../sam3
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if this is the actual sam3 package (has __init__.py and model_builder.py)
            init_file = os.path.join(path, "__init__.py")
            model_builder = os.path.join(path, "model_builder.py")
            if os.path.exists(init_file) or os.path.exists(model_builder):
                parent_path = os.path.dirname(path)
                if parent_path not in sys.path:
                    sys.path.insert(0, parent_path)
                return True
    return False


class SAM3Wrapper:
    """
    Wrapper class for SAM3 model operations.
    Handles model downloading, loading to GPU, and text prompt-based mask generation.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self._sam3_available = None
        self._import_error = None

    def is_sam3_available(self) -> bool:
        """Check if SAM3 package is installed and available."""
        if self._sam3_available is not None:
            return self._sam3_available

        # Try to setup local path first
        _setup_sam3_path()

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self._sam3_available = True
            self._import_error = None
        except ImportError as e:
            self._sam3_available = False
            self._import_error = str(e)
        except Exception as e:
            self._sam3_available = False
            self._import_error = str(e)

        return self._sam3_available

    def get_import_error(self) -> Optional[str]:
        """Get the import error message if SAM3 is not available."""
        return self._import_error

    def get_installation_instructions(self) -> str:
        """Return installation instructions for SAM3."""
        error_info = ""
        if self._import_error:
            error_info = f"\nImport Error: {self._import_error}\n"

        return f"""{error_info}
SAM3 Installation Instructions:

IMPORTANT: SAM3 requires Python 3.12 (not 3.13+)

1. Create a conda environment with Python 3.12:
   conda create -n sam3 python=3.12
   conda activate sam3

2. Install PyTorch with CUDA support:
   pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

3. Clone and install SAM3:
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .

4. Authenticate with Hugging Face (required for model download):
   huggingface-cli login

   Note: You need to request access to the SAM3 model on Hugging Face first:
   https://huggingface.co/facebook/sam3

5. Run this application from the sam3 conda environment:
   conda activate sam3
   python main.py
"""

    def check_gpu_available(self) -> Tuple[bool, str]:
        """
        Check if GPU (CUDA) is available.

        Returns:
            Tuple of (is_available, device_info)
        """
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, f"GPU: {device_name} ({memory_total:.1f} GB)"
            else:
                return False, "CUDA not available. Using CPU."
        except ImportError:
            return False, "PyTorch not installed."

    def download_model(self, progress_callback=None) -> Tuple[bool, str]:
        """
        Download SAM3 model from Hugging Face.
        The model is automatically downloaded when first loaded.

        Args:
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (success, message)
        """
        if not self.is_sam3_available():
            return False, "SAM3 is not installed. " + self.get_installation_instructions()

        try:
            if progress_callback:
                progress_callback("Checking Hugging Face authentication...")

            # Check if huggingface_hub is available and user is logged in
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                # This will raise an error if not authenticated
                api.whoami()
            except Exception as e:
                return False, f"Hugging Face authentication required. Run 'huggingface-cli login' first.\nError: {str(e)}"

            if progress_callback:
                progress_callback("Authentication verified. Model will be downloaded on first load.")

            return True, "Ready to download model. Model will be cached automatically on first load."

        except Exception as e:
            return False, f"Error checking model availability: {str(e)}"

    def load_model(self, use_gpu: bool = True, progress_callback=None) -> Tuple[bool, str]:
        """
        Load SAM3 model into memory.

        Args:
            use_gpu: Whether to load model on GPU (if available)
            progress_callback: Optional callback function for progress updates

        Returns:
            Tuple of (success, message)
        """
        if not self.is_sam3_available():
            return False, "SAM3 is not installed. " + self.get_installation_instructions()

        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            if progress_callback:
                progress_callback("Determining device...")

            # Determine device
            if use_gpu and torch.cuda.is_available():
                self.device = "cuda"
                device_info = f"GPU: {torch.cuda.get_device_name(0)}"
            else:
                self.device = "cpu"
                device_info = "CPU"

            if progress_callback:
                progress_callback(f"Loading SAM3 model on {device_info}...")

            # Build and load the model
            self.model = build_sam3_image_model()

            # Move model to device
            if self.device == "cuda":
                self.model = self.model.cuda()

            if progress_callback:
                progress_callback("Creating processor...")

            # Create processor
            self.processor = Sam3Processor(self.model)

            if progress_callback:
                progress_callback("Model loaded successfully!")

            return True, f"SAM3 model loaded successfully on {device_info}"

        except Exception as e:
            self.model = None
            self.processor = None
            return False, f"Error loading model: {str(e)}"

    def is_model_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.model is not None and self.processor is not None

    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        if self.model is not None:
            try:
                import torch
                del self.model
                del self.processor
                self.model = None
                self.processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def generate_mask_from_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        threshold: float = 0.5
    ) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], Optional[List[float]], str]:
        """
        Generate segmentation masks from image using text prompt.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            text_prompt: Text description of the object to segment (comma-separated for multiple prompts)
            threshold: Score threshold for filtering masks (0.0-1.0)

        Returns:
            Tuple of (combined_mask, individual_masks, scores, message)
            - combined_mask: Single mask combining all detected objects
            - individual_masks: List of individual masks for each detected object
            - scores: List of confidence scores for each mask
            - message: Status message
        """
        if not self.is_model_loaded():
            return None, None, None, "Model not loaded. Please load the model first."

        if image is None or len(image.shape) < 2:
            return None, None, None, "Invalid input image."

        if not text_prompt or not text_prompt.strip():
            return None, None, None, "Text prompt cannot be empty."

        try:
            import cv2

            # Convert BGR to RGB for PIL
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)

            # Split prompts by comma
            prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]

            if not prompts:
                return None, None, None, "Text prompt cannot be empty."

            all_individual_masks = []
            all_filtered_scores = []
            all_prompt_labels = []
            found_prompts = []

            for prompt in prompts:
                # Set image in processor
                inference_state = self.processor.set_image(pil_image)

                # Generate masks using text prompt
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=prompt
                )

                # Extract results
                masks = output.get("masks", [])
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])

                if len(masks) == 0:
                    continue

                # Convert masks to numpy arrays and filter by threshold
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    if hasattr(score, 'item'):
                        score = score.item()

                    if score >= threshold:
                        # Convert mask to numpy array if needed
                        if hasattr(mask, 'cpu'):
                            mask_np = mask.cpu().numpy()
                        elif hasattr(mask, 'numpy'):
                            mask_np = mask.numpy()
                        else:
                            mask_np = np.array(mask)

                        # Ensure mask is 2D binary
                        if len(mask_np.shape) > 2:
                            mask_np = mask_np.squeeze()

                        # Convert to uint8 binary mask
                        mask_np = (mask_np > 0.5).astype(np.uint8) * 255

                        all_individual_masks.append(mask_np)
                        all_filtered_scores.append(score)
                        all_prompt_labels.append(prompt)

                        if prompt not in found_prompts:
                            found_prompts.append(prompt)

            if len(all_individual_masks) == 0:
                return None, None, None, f"No objects found above threshold {threshold} for '{text_prompt}'."

            # Combine all masks into one
            combined_mask = np.zeros_like(all_individual_masks[0])
            for mask in all_individual_masks:
                combined_mask = np.maximum(combined_mask, mask)

            # Create result message
            if len(prompts) == 1:
                message = f"Found {len(all_individual_masks)} object(s) matching '{prompts[0]}'."
            else:
                message = f"Found {len(all_individual_masks)} object(s) for {len(found_prompts)}/{len(prompts)} prompts: {', '.join(found_prompts)}"

            return combined_mask, all_individual_masks, all_filtered_scores, message

        except Exception as e:
            return None, None, None, f"Error generating mask: {str(e)}"

    def create_colored_mask(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Create a colored version of a binary mask.

        Args:
            mask: Binary mask (0 and 255 values)
            color: BGR color tuple for the mask

        Returns:
            Colored mask as BGR image
        """
        h, w = mask.shape[:2]
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        mask_bool = mask > 127
        colored[mask_bool] = color

        return colored

    def overlay_mask_on_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay a mask on an image with transparency.

        Args:
            image: Original image (BGR)
            mask: Binary mask
            color: BGR color for the mask overlay
            alpha: Transparency of the overlay (0.0-1.0)

        Returns:
            Image with mask overlay
        """
        import cv2

        # Resize mask to match image if needed
        h, w = image.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Create colored mask
        colored_mask = self.create_colored_mask(mask, color)

        # Create output image
        output = image.copy()

        # Apply overlay where mask is non-zero
        mask_bool = mask > 127
        output[mask_bool] = cv2.addWeighted(
            image[mask_bool], 1 - alpha,
            colored_mask[mask_bool], alpha,
            0
        )

        return output


# Global instance for easy access
_sam3_instance = None


def get_sam3_wrapper() -> SAM3Wrapper:
    """Get the global SAM3 wrapper instance."""
    global _sam3_instance
    if _sam3_instance is None:
        _sam3_instance = SAM3Wrapper()
    return _sam3_instance
