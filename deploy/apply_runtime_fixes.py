"""Idempotent shared-expert fix for the pinned GLM SM120 v0.5.0 image."""
import ast
import hashlib
import pathlib

SOURCE = pathlib.Path('/opt/glm53/sources/sglang/python/sglang/srt/models/deepseek_v2.py')
OLD = '''            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and shared_gate_up_weight.dtype == torch.float8_e4m3fn
            )'''
NEW = '''            # A mixed shared MLP may have FP8 gate/up and native BF16 down.
            # FP8-only shared-expert shortcuts require both weights to be FP8.
            shared_down_weight = getattr(self.shared_experts.down_proj, "weight", None)
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and shared_gate_up_weight.dtype == torch.float8_e4m3fn
                and shared_down_weight is not None
                and shared_down_weight.dtype == torch.float8_e4m3fn
            )'''


def main():
    text = SOURCE.read_text()
    if NEW in text:
        print('Mixed shared-expert compatibility fix already present')
        return
    if text.count(OLD) != 1:
        raise RuntimeError('Unexpected runtime source; inspect before modifying it')
    backup = SOURCE.with_name(SOURCE.name+'.before-huihui-bf16-down')
    if not backup.exists():
        backup.write_text(text)
    updated = text.replace(OLD, NEW)
    ast.parse(updated)
    SOURCE.write_text(updated)
    print('Applied mixed shared-expert compatibility fix', hashlib.sha256(updated.encode()).hexdigest())


if __name__ == '__main__':
    main()
