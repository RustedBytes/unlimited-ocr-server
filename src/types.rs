mod inference;
mod job;
mod responses;
mod task;

pub use inference::{GenerationMetadata, InferenceMetadata, TensorMetadata};
pub use job::{JobRecord, JobStatus};
pub use responses::{
    ErrorResponse, GpuMemoryResponse, HealthResponse, MetricsResponse, QueueResponse,
    ReadinessResponse, WorkerHealth,
};
#[cfg(test)]
pub use task::{TaskPrompt, TaskType};
pub use task::{TaskSpec, TaskSpecError};

#[cfg(test)]
mod tests;
