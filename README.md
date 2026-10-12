# Assistive Desktop Navigator Pro

An advanced, hands-free operating system navigation engine designed to empower individuals with severe motor impairments. By utilizing standard video feeds from a generic webcam, this software-only solution translates facial metrics and high-fidelity hand kinematics into robust, real-time native OS mouse tracking, clicks, and document scrolling—eliminating the need for specialized or expensive depth-sensing hardware.

## 🚀 Key Features

*   **Asynchronous Dual-Engine Processing:** Built on a multi-threaded parallel architecture that isolates heavy AI rendering pipelines from video capture frames, guaranteeing a sustained 30+ FPS workspace without frozen UI states.
*   **Kinematic Safety Filters:** Employs an adaptive One-Euro damping algorithm combined with an absolute sub-pixel dead-zone check (rejecting micro-movements under 4 pixels or 3 degrees) to entirely ignore unintentional spasms, natural tremors, and tracking noise.
*   **Biometric Identity Firewall:** Features a built-in real-time security lock layer. Upon initialization, the system auto-enrolls the authorized user's primary facial symmetry profile vector grid. If an unauthorized face or intruder is detected in the frame, the cursor instantly freezes and blocks input injection until verification passes.
*   **Zero-Fatigue Workflow Triggers:**
    *   **Air Clutch Protocol:** A deliberate 0.5-second steady fist-hold securely locks or unlocks system cursor control tracking.
    *   **Dwell Selection:** Automated hover targeting that dispatches a rapid 0.20-second click when held steady inside an intentional 25-pixel hover cushion zone.
    *   **Wave Scrolling:** Integrated relative spatial tracking of finger landmarks for fluid, dynamic vertical web layout and document parsing.
