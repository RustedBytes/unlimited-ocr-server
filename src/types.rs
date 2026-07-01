mod inference;
mod job;
mod responses;
mod task;

pub use inference::{
    BoundingBox, GenerationMetadata, InferenceMetadata, OcrDetection, OcrResult, TensorMetadata,
};
pub use job::{JobRecord, JobStatus};
pub use responses::{
    ErrorResponse, GpuMemoryResponse, HealthResponse, MetricsResponse, PdfQueueResponse,
    QueueResponse, ReadinessResponse, SubmissionResponse, WorkerHealth,
};
pub use task::TaskSpec;

#[cfg(test)]
mod tests;
