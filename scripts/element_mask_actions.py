import ast
import data_utils as du
from abc import abstractmethod


class MaskAction:
    @abstractmethod
    def generate_masked_models(
        self, code: str
    ) -> list[str, str]:  # (code with mask, mask ori)
        raise NotImplementedError("To be implemented")


##################### args of keras.Model.compile #####################


# _MT_mask_keras_metric_arg = _mask_t('T_mask_keras_metric_arg')
class MetricMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_metric_arg.generator(codeast))
        # L... Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_learning_rate_arg = _mask_t('T_mask_keras_learning_rate_arg')
class LearningRateMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_learning_rate_arg.generator(codeast))
        # L... ('adam' -> __root__.keras.optimizers.Adam(<mask>)) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_optimizer_arg = _mask_t('T_mask_keras_optimizer_arg')
class OptimizerMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_optimizer_arg.generator(codeast))
        # L... ('<mask>', __root__.keras.<mask>) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_loss_arg = _mask_t('T_mask_keras_loss_arg')
class LossMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_loss_arg.generator(codeast))
        # L... ('<mask>', __root__.keras.<mask>) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


##################### args of keras.Model.fit #####################


# _MT_mask_keras_epochs_arg = _mask_t('T_mask_keras_epochs_arg')
class EpochsMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_epochs_arg.generator(codeast))
        # L...
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_batch_size_arg = _mask_t('T_mask_keras_batch_size_arg')
class BatchSizeMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_batch_size_arg.generator(codeast))
        # L...
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


##################### keras.layers.Layer / args of keras.layers.Layer #####################
# _MT_mask_keras_Layer = _mask_t('T_mask_keras_Layer')
class LayerMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_Layer.generator(codeast))
        # L... (...Layer(xxx,yyy) -> ...Layer(<mask0>)) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_activation_arg = _mask_t('T_mask_keras_activation_arg')
class ActivationMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_activation_arg.generator(codeast))
        # L... ('<mask>', __root__.keras.activations.<mask>) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


# _MT_mask_keras_initializer_arg = _mask_t('T_mask_keras_initializer_arg')
class InitializerMaskAction(MaskAction):
    def generate_masked_models(self, code: str) -> list[str, str]:
        result = []
        codeast = ast.parse(code)
        # L1 Template
        result.extend(du._MT_mask_keras_initializer_arg.generator(codeast))
        # L... ('<mask>', __root__.keras.initializers.<mask>) Ablation[2]
        return list(
            map(
                lambda x: (
                    ast.unparse(x.python_wmt_ast),
                    ast.unparse(x.masked_cfs_ast[0]),
                ),
                result,
            )
        )


def get_all_actions() -> list[MaskAction]:
    return sorted(
        [
            MetricMaskAction(),
            LearningRateMaskAction(),
            OptimizerMaskAction(),
            LossMaskAction(),
            EpochsMaskAction(),
            BatchSizeMaskAction(),
            LayerMaskAction(),
            ActivationMaskAction(),
            InitializerMaskAction(),
        ],
        key=lambda x: x.__class__.__name__,
    )
