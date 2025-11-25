# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.entrypoints.cli.serve import ServeSubcommand
from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.cli_args import (
    validate_parsed_serve_args as _validate_args,
)
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.logger import init_logger

logger = init_logger(__name__)

DESCRIPTION = """Launch a prefill-only OpenAI-compatible API server.

This mode is tailored for hybrid prefilling: every request is treated as a
single chunk and the server enforces `max_tokens=1`.
"""


class PrefillSubcommand(CLISubcommand):
    """The `prefill` subcommand for the vLLM CLI."""

    name = "prefill"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        # Mark the run as prefill-only.
        args.prefill_mode = True

        logger.info(
            "Starting vLLM in prefill mode: the server will require "
            "`max_tokens=1` for every request."
        )

        ServeSubcommand.cmd(args)

    def validate(self, args: argparse.Namespace) -> None:
        _validate_args(args)

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        prefill_parser = subparsers.add_parser(
            self.name,
            description=DESCRIPTION,
            usage="vllm prefill [model_tag] [options]",
        )
        prefill_parser = make_arg_parser(prefill_parser)
        prefill_parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return prefill_parser


def cmd_init() -> list[CLISubcommand]:
    return [PrefillSubcommand()]
