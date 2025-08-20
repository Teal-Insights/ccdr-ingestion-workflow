import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator


def file_starts_with(path: Path, prefix: str, encoding: str = "utf-8") -> bool:
    """Memory-efficient helper to check if a file starts with a given prefix"""
    prefix_bytes = prefix.encode(encoding)
    try:
        with path.open("rb") as f:
            return f.read(len(prefix_bytes)) == prefix_bytes
    except FileNotFoundError:
        return False


class GetTextFileContentsRequest(BaseModel):
    """Request model for getting text file contents."""

    file_path: str = Field(..., description="Path to the text file")
    start: int = Field(1, description="Starting line number (1-based)")
    end: Optional[int] = Field(None, description="Ending line number (inclusive)")


class GetTextFileContentsResponse(BaseModel):
    """Response model for getting text file contents."""

    contents: str = Field(..., description="File contents")
    start: int = Field(..., description="Starting line number")
    end: int = Field(..., description="Ending line number")
    hash: str = Field(..., description="Hash of the contents")


class EditPatch(BaseModel):
    """Model for a single edit patch operation."""

    start: int = Field(1, description="Starting line for edit")
    end: Optional[int] = Field(None, description="Ending line for edit")
    contents: str = Field(..., description="New content to insert")
    range_hash: Optional[str] = Field(
        None,  # None for new patches, must be explicitly set
        description="Hash of content being replaced. Empty string for insertions.",
    )

    @model_validator(mode="after")
    def validate_range_hash(self) -> "EditPatch":
        """Validate that range_hash is set and handle end field validation."""
        # range_hash must be explicitly set
        if self.range_hash is None:
            raise ValueError("range_hash is required")

        # For safety, convert None to the special range hash value
        if self.end is None and self.range_hash != "":
            # Special case: patch with end=None is allowed
            pass

        return self


class EditFileOperation(BaseModel):
    """Model for individual file edit operation."""

    path: str = Field(..., description="Path to the file")
    hash: str = Field(..., description="Hash of original contents")
    patches: List[EditPatch] = Field(..., description="Edit operations to apply")


class EditResult(BaseModel):
    """Model for edit operation result."""

    result: str = Field(..., description="Operation result (ok/error)")
    reason: Optional[str] = Field(None, description="Error message if applicable")
    hash: Optional[str] = Field(
        None, description="Current content hash (None for missing files)"
    )

    @model_validator(mode="after")
    def validate_error_result(self) -> "EditResult":
        """Remove hash when result is error."""
        if self.result == "error":
            object.__setattr__(self, "hash", None)
        return self

    def to_dict(self) -> Dict:
        """Convert EditResult to a dictionary."""
        result = {"result": self.result}
        if self.reason is not None:
            result["reason"] = self.reason
        if self.hash is not None:
            result["hash"] = self.hash
        return result


class EditTextFileContentsRequest(BaseModel):
    """Request model for editing text file contents.

    Example:
    {
        "files": [
            {
                "path": "/path/to/file",
                "hash": "abc123...",
                "patches": [
                    {
                        "start": 1,  # default: 1 (top of file)
                        "end": null,  # default: null (end of file)
                        "contents": "new content"
                    }
                ]
            }
        ]
    }
    """

    files: List[EditFileOperation] = Field(..., description="List of file operations")


class FileRange(BaseModel):
    """Represents a line range in a file."""

    start: int = Field(..., description="Starting line number (1-based)")
    end: Optional[int] = Field(
        None, description="Ending line number (null for end of file)"
    )
    range_hash: Optional[str] = Field(
        None, description="Hash of the content to be deleted"
    )


class FileRanges(BaseModel):
    """Represents a file and its line ranges."""

    file_path: str = Field(..., description="Path to the text file")
    ranges: List[FileRange] = Field(
        ..., description="List of line ranges to read from the file"
    )


class InsertTextFileContentsRequest(BaseModel):
    """Request model for inserting text into a file.

    Example:
    {
        "path": "/path/to/file",
        "file_hash": "abc123...",
        "after": 5,  # Insert after line 5
        "contents": "new content"
    }
    or
    {
        "path": "/path/to/file",
        "file_hash": "abc123...",
        "before": 5,  # Insert before line 5
        "contents": "new content"
    }
    """

    path: str = Field(..., description="Path to the text file")
    file_hash: str = Field(..., description="Hash of original contents")
    after: Optional[int] = Field(
        None, description="Line number after which to insert content"
    )
    before: Optional[int] = Field(
        None, description="Line number before which to insert content"
    )
    encoding: Optional[str] = Field(
        "utf-8", description="Text encoding (default: 'utf-8')"
    )
    contents: str = Field(..., description="Content to insert")

    @model_validator(mode="after")
    def validate_position(self) -> "InsertTextFileContentsRequest":
        """Validate that exactly one of after or before is specified."""
        if (self.after is None and self.before is None) or (
            self.after is not None and self.before is not None
        ):
            raise ValueError("Exactly one of 'after' or 'before' must be specified")
        return self

    @field_validator("after", "before")
    def validate_line_number(cls, v) -> Optional[int]:
        """Validate that line numbers are positive."""
        if v is not None and v < 1:
            raise ValueError("Line numbers must be positive")
        return v


class DeleteTextFileContentsRequest(BaseModel):
    """Request model for deleting text from a file.
    Example:
    {
        "file_path": "/path/to/file",
        "file_hash": "abc123...",
        "ranges": [
            {
                "start": 5,
                "end": 10,
                "range_hash": "def456..."
            }
        ]
    }
    """

    file_path: str = Field(..., description="Path to the text file")
    file_hash: str = Field(..., description="Hash of original contents")
    ranges: List[FileRange] = Field(..., description="List of ranges to delete")
    encoding: Optional[str] = Field(
        "utf-8", description="Text encoding (default: 'utf-8')"
    )


class PatchTextFileContentsRequest(BaseModel):
    """Request model for patching text in a file.
    Example:
    {
        "file_path": "/path/to/file",
        "file_hash": "abc123...",
        "patches": [
            {
                "start": 5,
                "end": 10,
                "contents": "new content",
                "range_hash": "def456..."
            }
        ]
    }
    """

    file_path: str = Field(..., description="Path to the text file")
    file_hash: str = Field(..., description="Hash of original contents")
    patches: List[EditPatch] = Field(..., description="List of patches to apply")
    encoding: Optional[str] = Field(
        "utf-8", description="Text encoding (default: 'utf-8')"
    )


class TextEditorService:
    """Service class for text file operations."""

    @staticmethod
    def calculate_hash(content: str) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def read_file_contents(
        file_path: str, start: int = 1, end: Optional[int] = None
    ) -> Tuple[str, int, int]:
        """Read file contents within specified line range."""
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Adjust line numbers to 0-based index
        start = max(1, start) - 1
        end = len(lines) if end is None else min(end, len(lines))

        selected_lines = lines[start:end]
        content = "".join(selected_lines)

        return content, start + 1, end

    @staticmethod
    def validate_patches(patches: List[EditPatch], total_lines: int) -> bool:
        """Validate patches for overlaps and bounds."""
        # Sort patches by start
        sorted_patches = sorted(patches, key=lambda x: x.start)

        prev_end = 0
        for patch in sorted_patches:
            if patch.start <= prev_end:
                return False
            patch_end = patch.end or total_lines
            if patch_end > total_lines:
                return False
            prev_end = patch_end

        return True

    def edit_file_contents(
        self, file_path: str, operation: EditFileOperation
    ) -> Dict[str, EditResult]:
        """Edit file contents with conflict detection."""
        current_hash = None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                current_content = f.read()
                current_hash = self.calculate_hash(current_content)

            # Check for conflicts
            if current_hash != operation.hash:
                return {
                    file_path: EditResult(
                        result="error",
                        reason="Content hash mismatch",
                        hash=current_hash,
                    )
                }

            # Split content into lines
            lines = current_content.splitlines(keepends=True)

            # Validate patches
            if not self.validate_patches(operation.patches, len(lines)):
                return {
                    file_path: EditResult(
                        result="error",
                        reason="Invalid patch ranges",
                        hash=current_hash,
                    )
                }

            # Apply patches
            new_lines = lines.copy()
            for patch in operation.patches:
                start_idx = patch.start - 1
                end_idx = patch.end if patch.end else len(lines)
                patch_lines = patch.contents.splitlines(keepends=True)
                new_lines[start_idx:end_idx] = patch_lines

            # Write new content
            new_content = "".join(new_lines)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            new_hash = self.calculate_hash(new_content)
            return {
                file_path: EditResult(
                    result="ok",
                    hash=new_hash,
                    reason=None,
                )
            }

        except FileNotFoundError as e:
            return {
                file_path: EditResult(
                    result="error",
                    reason=str(e),
                    hash=None,
                )
            }
        except Exception as e:
            return {
                file_path: EditResult(
                    result="error",
                    reason=str(e),
                    hash=None,  # Don't return the current hash on error
                )
            }

    def delete_text_file_contents(
        self,
        request: DeleteTextFileContentsRequest,
    ) -> Dict[str, EditResult]:
        """Delete specified ranges from a text file with conflict detection."""
        current_hash = None
        try:
            with open(request.file_path, "r", encoding=request.encoding) as f:
                current_content = f.read()
                current_hash = self.calculate_hash(current_content)

            # Check for conflicts
            if current_hash != request.file_hash:
                return {
                    request.file_path: EditResult(
                        result="error",
                        reason="Content hash mismatch",
                        hash=current_hash,
                    )
                }

            # Split content into lines
            lines = current_content.splitlines(keepends=True)

            # Validate ranges
            if not request.ranges:  # Check for empty ranges list
                return {
                    request.file_path: EditResult(
                        result="error",
                        reason="Missing required argument: ranges",
                        hash=current_hash,
                    )
                }

            if not self.validate_ranges(request.ranges, len(lines)):
                return {
                    request.file_path: EditResult(
                        result="error",
                        reason="Invalid ranges",
                        hash=current_hash,
                    )
                }

            # Delete ranges in reverse order to handle line number shifts
            new_lines = lines.copy()
            sorted_ranges = sorted(request.ranges, key=lambda x: x.start, reverse=True)
            for range_ in sorted_ranges:
                start_idx = range_.start - 1
                end_idx = range_.end if range_.end else len(lines)
                target_content = "".join(lines[start_idx:end_idx])
                target_hash = self.calculate_hash(target_content)
                if target_hash != range_.range_hash:
                    return {
                        request.file_path: EditResult(
                            result="error",
                            reason=f"Content hash mismatch for range {range_.start}-{range_.end}",
                            hash=current_hash,
                        )
                    }
                del new_lines[start_idx:end_idx]

            # Write new content
            new_content = "".join(new_lines)
            with open(request.file_path, "w", encoding=request.encoding) as f:
                f.write(new_content)

            new_hash = self.calculate_hash(new_content)
            return {
                request.file_path: EditResult(
                    result="ok",
                    hash=new_hash,
                    reason=None,
                )
            }

        except FileNotFoundError as e:
            return {
                request.file_path: EditResult(
                    result="error",
                    reason=str(e),
                    hash=None,
                )
            }
        except Exception as e:
            return {
                request.file_path: EditResult(
                    result="error",
                    reason=f"Error deleting contents: {str(e)}",
                    hash=None,
                )
            }

    @staticmethod
    def validate_ranges(ranges: List[FileRange], total_lines: int) -> bool:
        """Validate ranges for overlaps and bounds."""
        # Sort ranges by start line
        sorted_ranges = sorted(ranges, key=lambda x: x.start)

        prev_end = 0
        for range_ in sorted_ranges:
            if range_.start <= prev_end:
                return False  # Overlapping ranges
            if range_.start < 1:
                return False  # Invalid start line
            range_end = range_.end or total_lines
            if range_end > total_lines:
                return False  # Exceeding file length
            if range_.end is not None and range_.end < range_.start:
                return False  # End before start
            prev_end = range_end

        return True