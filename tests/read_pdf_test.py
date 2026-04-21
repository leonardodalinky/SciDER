"""Tests for PDF-reading support in the Read tool.

The Read tool must:
- Return a ``ToolResult`` carrying a ``ToolDocument`` with base64-encoded
  PDF bytes for small-enough PDFs.
- Fall back to text-only metadata for PDFs above ``MAX_PDF_PAYLOAD_BYTES``.
- Emit the filename on the document so providers that display it (OpenAI
  Responses API) can show it to the model.
"""

from __future__ import annotations

import base64
from pathlib import Path

from scider.core.llms import _serialize_messages_for_provider, _serialize_messages_for_responses_api
from scider.core.types import Message
from scider.tools.base import ToolContext, ToolResult
from scider.tools.fs.read_file import MAX_PDF_PAYLOAD_BYTES, ReadFileTool

_MINIMAL_PDF = (
    b"%PDF-1.1\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000009 00000 n \n0000000052 00000 n \n0000000099 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n152\n%%EOF\n"
)


def _ctx() -> ToolContext:
    return ToolContext(agent_name="test")


class TestPdfRead:
    def test_small_pdf_returns_tool_result_with_document(self, tmp_path: Path):
        p = tmp_path / "tiny.pdf"
        p.write_bytes(_MINIMAL_PDF)

        result = ReadFileTool().call(_ctx(), file_path=str(p))
        assert isinstance(result, ToolResult)
        assert len(result.documents) == 1
        doc = result.documents[0]
        assert doc.media_type == "application/pdf"
        assert doc.filename == "tiny.pdf"
        # Round-trip base64 → bytes matches original.
        assert base64.b64decode(doc.data) == _MINIMAL_PDF
        assert "PDF attached" in result.text

    def test_oversize_pdf_falls_back_to_metadata(self, tmp_path: Path, monkeypatch):
        # Pretend the file exceeds the limit without actually writing 20 MB.
        p = tmp_path / "big.pdf"
        p.write_bytes(_MINIMAL_PDF)

        import scider.tools.fs.read_file as rf

        monkeypatch.setattr(rf, "MAX_PDF_PAYLOAD_BYTES", len(_MINIMAL_PDF) - 1)

        result = ReadFileTool().call(_ctx(), file_path=str(p))
        assert isinstance(result, str)
        assert "metadata only" in result
        assert "pdf" in result


class TestProviderSerialization:
    def _tool_msg_with_pdf(self) -> Message:
        return Message(
            role="tool",
            tool_call_id="call_pdf_1",
            tool_name="Read",
            content="[File: /x/doc.pdf]\n[Size: 1234 bytes]",
            tool_result_documents=[
                {
                    "media_type": "application/pdf",
                    "data": base64.b64encode(_MINIMAL_PDF).decode("ascii"),
                    "filename": "doc.pdf",
                }
            ],
        )

    def test_anthropic_gets_inline_file_block(self):
        out = _serialize_messages_for_provider([self._tool_msg_with_pdf()], "claude-sonnet-4-6")
        assert len(out) == 1
        msg = out[0]
        assert msg["role"] == "tool"
        blocks = msg["content"]
        file_blocks = [b for b in blocks if b.get("type") == "file"]
        assert len(file_blocks) == 1
        assert file_blocks[0]["file"]["filename"] == "doc.pdf"
        assert file_blocks[0]["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_gemini_gets_inline_file_block(self):
        out = _serialize_messages_for_provider([self._tool_msg_with_pdf()], "gemini/gemini-2.5-pro")
        assert any(b.get("type") == "file" for b in out[0]["content"])

    def test_openai_chat_gets_textual_note_only(self):
        out = _serialize_messages_for_provider([self._tool_msg_with_pdf()], "gpt-5-mini")
        # Two entries: tool message (text) + synthetic user note
        assert len(out) == 2
        assert out[0]["role"] == "tool"
        assert out[1]["role"] == "user"
        # No image/file blocks — just text notes
        for block in out[1]["content"]:
            assert block["type"] == "text"
        assert any("not viewable" in b["text"] for b in out[1]["content"])

    def test_openai_responses_api_gets_input_file(self):
        out = _serialize_messages_for_responses_api([self._tool_msg_with_pdf()])
        # function_call_output followed by synthetic user message
        assert any(entry.get("type") == "message" and entry.get("role") == "user" for entry in out)
        user_msg = next(e for e in out if e.get("type") == "message" and e.get("role") == "user")
        input_files = [b for b in user_msg["content"] if b.get("type") == "input_file"]
        assert len(input_files) == 1
        assert input_files[0]["filename"] == "doc.pdf"
        assert input_files[0]["file_data"].startswith("data:application/pdf;base64,")


def test_description_mentions_pdf():
    """Tool description should advertise the new PDF capability."""
    tool = ReadFileTool()
    assert "PDF" in tool.description
    assert str(MAX_PDF_PAYLOAD_BYTES // (1024 * 1024)) in tool.description
