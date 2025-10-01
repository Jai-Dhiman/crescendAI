import { z } from 'zod';

// User validation schemas
export const userSchema = z.object({
  id: z.string(),
  email: z.string().email(),
  name: z.string().min(1),
  avatar: z.string().url().optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
});

// Auth validation schemas
export const signUpSchema = z.object({
  name: z.string().min(1, "Name is required"),
  email: z.string().email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
  confirmPassword: z.string(),
}).refine(data => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

export const signInSchema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(1, "Password is required"),
});

// Recording validation schemas
export const recordingUploadSchema = z.object({
  title: z.string().min(1, "Title is required"),
  description: z.string().optional(),
  file: z.any().refine(
    (file) => {
      if (!file) return false;
      const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4'];
      return validTypes.includes(file.type);
    },
    "Please select a valid audio file (WAV, MP3, or M4A)"
  ).refine(
    (file) => {
      if (!file) return false;
      const maxSize = 50 * 1024 * 1024; // 50MB
      return file.size <= maxSize;
    },
    "File size must be less than 50MB"
  ),
});

// Audio validation helpers
export const validateAudioFile = (file: File | null): string | null => {
  if (!file) return "No file selected";
  
  const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4'];
  if (!validTypes.includes(file.type)) {
    return "Please select a valid audio file (WAV, MP3, or M4A)";
  }
  
  const maxSize = 50 * 1024 * 1024; // 50MB
  if (file.size > maxSize) {
    return "File size must be less than 50MB";
  }
  
  return null;
};

export const validateAudioDuration = (duration: number): string | null => {
  if (duration < 30) return "Recording must be at least 30 seconds";
  if (duration > 180) return "Recording must be less than 3 minutes";
  return null;
};