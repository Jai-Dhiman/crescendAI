/**
 * Secure File Upload Validation for CrescendAI
 * 
 * Provides comprehensive security validation for audio file uploads
 * including MIME type checking, file size limits, and malware detection.
 */

import { clientConfig } from '../config/environment';

interface FileValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  sanitizedFilename?: string;
  fileMetadata?: FileMetadata;
}

interface FileMetadata {
  originalName: string;
  sanitizedName: string;
  size: number;
  type: string;
  extension: string;
  lastModified: number;
  hash?: string;
}

interface FileSecurityConfig {
  maxFileSize: number;
  allowedMimeTypes: readonly string[];
  allowedExtensions: readonly string[];
  blockedExtensions: readonly string[];
  maxFilenameLength: number;
  requireAudioHeader: boolean;
}

/**
 * Security configuration for file uploads
 */
const SECURITY_CONFIG: FileSecurityConfig = {
  maxFileSize: clientConfig.maxFileSize,
  
  // Allowed MIME types for audio files
  allowedMimeTypes: clientConfig.allowedFileTypes,
  
  // Allowed file extensions
  allowedExtensions: [
    '.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac',
    '.opus', '.wma', '.ape', '.wv', '.tta'
  ],
  
  // Blocked extensions (security risk)
  blockedExtensions: [
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
    '.js', '.jar', '.app', '.deb', '.pkg', '.dmg',
    '.iso', '.img', '.vbs', '.ps1', '.sh'
  ],
  
  maxFilenameLength: 255,
  requireAudioHeader: true,
} as const;

/**
 * Audio file magic number signatures for header validation
 */
const AUDIO_SIGNATURES: Record<string, Uint8Array[]> = {
  'audio/mpeg': [
    new Uint8Array([0xFF, 0xFB]), // MP3 (MPEG-1 Layer 3)
    new Uint8Array([0xFF, 0xF3]), // MP3 (MPEG-2 Layer 3)
    new Uint8Array([0xFF, 0xF2]), // MP3 (MPEG-2.5 Layer 3)
    new Uint8Array([0x49, 0x44, 0x33]), // ID3 tag
  ],
  'audio/wav': [
    new Uint8Array([0x52, 0x49, 0x46, 0x46]), // RIFF header
  ],
  'audio/x-wav': [
    new Uint8Array([0x52, 0x49, 0x46, 0x46]), // RIFF header (same as audio/wav)
  ],
  'audio/ogg': [
    new Uint8Array([0x4F, 0x67, 0x67, 0x53]), // OggS
  ],
  'audio/flac': [
    new Uint8Array([0x66, 0x4C, 0x61, 0x43]), // fLaC
  ],
  'audio/mp4': [
    new Uint8Array([0x00, 0x00, 0x00, 0x20, 0x66, 0x74, 0x79, 0x70]), // ftyp (MP4)
    new Uint8Array([0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70]), // ftyp (MP4)
  ],
  'audio/aac': [
    new Uint8Array([0xFF, 0xF1]), // AAC ADTS
    new Uint8Array([0xFF, 0xF9]), // AAC ADTS
  ],
};

/**
 * Sanitize filename to prevent path traversal and injection attacks
 */
function sanitizeFilename(filename: string): string {
  // Remove or replace dangerous characters
  let sanitized = filename
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_') // Replace illegal chars with underscore
    .replace(/^\.+/, '') // Remove leading dots
    .replace(/\.+$/, '') // Remove trailing dots
    .replace(/\s+/g, '_') // Replace spaces with underscores
    .replace(/_+/g, '_') // Collapse multiple underscores
    .toLowerCase(); // Convert to lowercase

  // Ensure filename isn't too long
  if (sanitized.length > SECURITY_CONFIG.maxFilenameLength) {
    const extension = sanitized.substring(sanitized.lastIndexOf('.'));
    const nameWithoutExt = sanitized.substring(0, sanitized.lastIndexOf('.'));
    const maxNameLength = SECURITY_CONFIG.maxFilenameLength - extension.length;
    sanitized = nameWithoutExt.substring(0, maxNameLength) + extension;
  }

  // Ensure filename isn't empty
  if (!sanitized || sanitized === '_') {
    sanitized = `audio_${Date.now()}.mp3`;
  }

  return sanitized;
}

/**
 * Extract file extension safely
 */
function getFileExtension(filename: string): string {
  const lastDotIndex = filename.lastIndexOf('.');
  if (lastDotIndex === -1) return '';
  return filename.substring(lastDotIndex).toLowerCase();
}

/**
 * Validate file extension against allowed/blocked lists
 */
function validateExtension(filename: string): { isValid: boolean; error?: string } {
  const extension = getFileExtension(filename);
  
  if (!extension) {
    return { isValid: false, error: 'File must have an extension' };
  }

  // Check against blocked extensions first
  if (SECURITY_CONFIG.blockedExtensions.includes(extension)) {
    return { isValid: false, error: `File extension "${extension}" is not allowed for security reasons` };
  }

  // Check against allowed extensions
  if (!SECURITY_CONFIG.allowedExtensions.includes(extension)) {
    return { isValid: false, error: `File extension "${extension}" is not supported. Allowed: ${SECURITY_CONFIG.allowedExtensions.join(', ')}` };
  }

  return { isValid: true };
}

/**
 * Validate MIME type
 */
function validateMimeType(file: File): { isValid: boolean; error?: string } {
  if (!file.type) {
    return { isValid: false, error: 'File MIME type is missing' };
  }

  if (!SECURITY_CONFIG.allowedMimeTypes.includes(file.type)) {
    return { 
      isValid: false, 
      error: `MIME type "${file.type}" is not allowed. Allowed: ${SECURITY_CONFIG.allowedMimeTypes.join(', ')}` 
    };
  }

  return { isValid: true };
}

/**
 * Validate file size
 */
function validateFileSize(file: File): { isValid: boolean; error?: string } {
  if (file.size <= 0) {
    return { isValid: false, error: 'File appears to be empty' };
  }

  if (file.size > SECURITY_CONFIG.maxFileSize) {
    const maxSizeMB = Math.round(SECURITY_CONFIG.maxFileSize / (1024 * 1024));
    const fileSizeMB = Math.round(file.size / (1024 * 1024));
    return { 
      isValid: false, 
      error: `File size (${fileSizeMB}MB) exceeds maximum allowed size (${maxSizeMB}MB)` 
    };
  }

  return { isValid: true };
}

/**
 * Read file header for magic number validation
 */
async function readFileHeader(file: File, bytesToRead: number = 16): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = () => {
      const arrayBuffer = reader.result as ArrayBuffer;
      resolve(new Uint8Array(arrayBuffer));
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file header'));
    };
    
    // Read only the first few bytes
    const blob = file.slice(0, bytesToRead);
    reader.readAsArrayBuffer(blob);
  });
}

/**
 * Check if byte arrays match
 */
function bytesMatch(signature: Uint8Array, header: Uint8Array): boolean {
  if (signature.length > header.length) return false;
  
  for (let i = 0; i < signature.length; i++) {
    if (signature[i] !== header[i]) return false;
  }
  
  return true;
}

/**
 * Validate file header against known audio signatures
 */
async function validateFileHeader(file: File): Promise<{ isValid: boolean; error?: string }> {
  try {
    const header = await readFileHeader(file, 32);
    const signatures = AUDIO_SIGNATURES[file.type] || [];
    
    if (signatures.length === 0) {
      return { isValid: true }; // No signatures defined for this type
    }

    // Check if any signature matches
    const matches = signatures.some(signature => bytesMatch(signature, header));
    
    if (!matches) {
      return { 
        isValid: false, 
        error: `File header does not match expected format for ${file.type}. This may indicate file corruption or incorrect file type.` 
      };
    }

    return { isValid: true };
  } catch (error) {
    return { 
      isValid: false, 
      error: `Failed to validate file header: ${error instanceof Error ? error.message : 'Unknown error'}` 
    };
  }
}

/**
 * Check for common file upload attack patterns
 */
function detectMaliciousPatterns(filename: string): string[] {
  const warnings: string[] = [];
  
  // Check for path traversal attempts
  if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
    warnings.push('Filename contains path traversal characters');
  }

  // Check for script injection attempts
  if (/\.(php|jsp|asp|js|html|htm)$/i.test(filename)) {
    warnings.push('Filename has script-like extension');
  }

  // Check for hidden file indicators
  if (filename.startsWith('.')) {
    warnings.push('Filename starts with dot (hidden file)');
  }

  // Check for excessively long filename
  if (filename.length > 100) {
    warnings.push('Filename is unusually long');
  }

  return warnings;
}

/**
 * Generate a secure hash of the file (for deduplication and integrity)
 */
async function generateFileHash(file: File): Promise<string> {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  } catch (error) {
    console.warn('Failed to generate file hash:', error);
    return '';
  }
}

/**
 * Create file metadata object
 */
function createFileMetadata(file: File, sanitizedFilename: string): FileMetadata {
  return {
    originalName: file.name,
    sanitizedName: sanitizedFilename,
    size: file.size,
    type: file.type,
    extension: getFileExtension(sanitizedFilename),
    lastModified: file.lastModified,
  };
}

/**
 * Comprehensive file validation
 */
export async function validateUploadFile(file: File): Promise<FileValidationResult> {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Basic file checks
  if (!file) {
    return {
      isValid: false,
      errors: ['No file provided'],
      warnings: [],
    };
  }

  // File size validation
  const sizeValidation = validateFileSize(file);
  if (!sizeValidation.isValid) {
    errors.push(sizeValidation.error!);
  }

  // MIME type validation
  const mimeValidation = validateMimeType(file);
  if (!mimeValidation.isValid) {
    errors.push(mimeValidation.error!);
  }

  // Filename validation
  const extensionValidation = validateExtension(file.name);
  if (!extensionValidation.isValid) {
    errors.push(extensionValidation.error!);
  }

  // Security pattern detection
  const securityWarnings = detectMaliciousPatterns(file.name);
  warnings.push(...securityWarnings);

  // Sanitize filename
  const sanitizedFilename = sanitizeFilename(file.name);

  // File header validation (if no errors so far)
  if (errors.length === 0 && SECURITY_CONFIG.requireAudioHeader) {
    const headerValidation = await validateFileHeader(file);
    if (!headerValidation.isValid) {
      errors.push(headerValidation.error!);
    }
  }

  // Generate file metadata
  let fileMetadata: FileMetadata | undefined;
  let hash: string | undefined;
  
  if (errors.length === 0) {
    fileMetadata = createFileMetadata(file, sanitizedFilename);
    hash = await generateFileHash(file);
    if (hash) {
      fileMetadata.hash = hash;
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    sanitizedFilename,
    fileMetadata,
  };
}

/**
 * Rate limiting for file uploads (client-side)
 */
class UploadRateLimit {
  private uploadTimes: number[] = [];
  private readonly maxUploads: number;
  private readonly timeWindow: number;

  constructor(maxUploads = 10, timeWindowMs = 60000) {
    this.maxUploads = maxUploads;
    this.timeWindow = timeWindowMs;
  }

  canUpload(): { allowed: boolean; error?: string } {
    const now = Date.now();
    
    // Remove old entries
    this.uploadTimes = this.uploadTimes.filter(time => now - time < this.timeWindow);
    
    if (this.uploadTimes.length >= this.maxUploads) {
      const oldestUpload = Math.min(...this.uploadTimes);
      const waitTime = Math.ceil((this.timeWindow - (now - oldestUpload)) / 1000);
      return {
        allowed: false,
        error: `Upload rate limit exceeded. Please wait ${waitTime} seconds before uploading again.`,
      };
    }

    // Record this upload attempt
    this.uploadTimes.push(now);
    return { allowed: true };
  }

  reset(): void {
    this.uploadTimes = [];
  }
}

// Export singleton rate limiter
export const uploadRateLimit = new UploadRateLimit(
  clientConfig.rateLimitMaxRequests,
  clientConfig.rateLimitWindow
);

/**
 * Secure file upload preparation
 */
export async function prepareSecureUpload(file: File): Promise<{
  isValid: boolean;
  file?: File;
  metadata?: FileMetadata;
  errors: string[];
  warnings: string[];
}> {
  // Check rate limiting
  const rateLimitCheck = uploadRateLimit.canUpload();
  if (!rateLimitCheck.allowed) {
    return {
      isValid: false,
      errors: [rateLimitCheck.error!],
      warnings: [],
    };
  }

  // Validate file
  const validation = await validateUploadFile(file);
  if (!validation.isValid) {
    return {
      isValid: false,
      errors: validation.errors,
      warnings: validation.warnings,
    };
  }

  // Create a new File object with sanitized name if needed
  let processedFile = file;
  if (validation.sanitizedFilename && validation.sanitizedFilename !== file.name) {
    processedFile = new File([file], validation.sanitizedFilename, {
      type: file.type,
      lastModified: file.lastModified,
    });
  }

  return {
    isValid: true,
    file: processedFile,
    metadata: validation.fileMetadata,
    errors: [],
    warnings: validation.warnings,
  };
}

/**
 * Export configuration for external use
 */
export { SECURITY_CONFIG as fileSecurityConfig };