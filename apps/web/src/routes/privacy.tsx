import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/privacy")({ component: PrivacyPage });

// TODO: Update entity name and contact details once incorporated
const ENTITY_NAME = "Crescend";
const EFFECTIVE_DATE = "March 9, 2026";

function PrivacyPage() {
	return (
		<div className="pt-24 pb-16 lg:pb-24">
			<div className="max-w-3xl mx-auto px-6 lg:px-12">
				<h1 className="font-display text-display-md text-cream mb-4">
					Privacy Policy
				</h1>
				<p className="text-body-sm text-text-tertiary mb-12">
					Effective: {EFFECTIVE_DATE}
				</p>

				<div className="space-y-10 text-body-md text-text-secondary leading-relaxed">
					<Section title="1. Information We Collect">
						<p className="mb-4">
							We collect the following types of information when you use{" "}
							{ENTITY_NAME}:
						</p>

						<h3 className="font-display text-cream text-body-md mb-2">
							Account Information
						</h3>
						<p className="mb-4">
							When you sign in with Apple, we receive your Apple user identifier
							and, if you choose to share them, your name and email address.
							Apple may provide a private relay email address.
						</p>

						<h3 className="font-display text-cream text-body-md mb-2">
							Audio Recordings
						</h3>
						<p className="mb-4">
							When you use the recording feature, we capture audio of your piano
							playing. On iOS, audio is processed on your device using on-device
							machine learning models. On the web, audio chunks are sent to our
							servers for processing and are not retained after analysis is
							complete.
						</p>

						<h3 className="font-display text-cream text-body-md mb-2">
							Practice Data
						</h3>
						<p className="mb-4">
							We store practice session metadata, including timestamps, duration,
							piece information, and AI-generated observations about your
							playing. This data is used to track your progress and personalize
							feedback.
						</p>

						<h3 className="font-display text-cream text-body-md mb-2">
							Usage Data
						</h3>
						<p>
							We collect standard usage data such as device type, operating
							system, app version, and interaction patterns to improve the
							Service.
						</p>
					</Section>

					<Section title="2. How We Use Your Information">
						<ul className="list-disc pl-6 space-y-2">
							<li>
								Provide and improve the Service, including generating
								personalized practice feedback
							</li>
							<li>
								Process audio recordings through our machine learning models to
								analyze musical expression
							</li>
							<li>
								Maintain your practice history and track your progress over
								time
							</li>
							<li>
								Communicate with you about the Service, including updates and
								support
							</li>
							<li>
								Detect and prevent fraud, abuse, or security issues
							</li>
						</ul>
					</Section>

					<Section title="3. Audio Data Handling">
						<p>
							Your audio recordings are handled with particular care:
						</p>
						<ul className="list-disc pl-6 mt-3 space-y-2">
							<li>
								<strong className="text-cream">iOS:</strong> Audio is processed
								entirely on your device using Core ML. Raw audio data is not
								sent to our servers unless you explicitly opt in to cloud
								features.
							</li>
							<li>
								<strong className="text-cream">Web:</strong> Audio chunks are
								transmitted to our inference servers over encrypted connections
								(TLS). Audio data is processed in real time and is not stored
								after analysis.
							</li>
							<li>
								We do not sell, license, or share your audio recordings with
								third parties.
							</li>
							<li>
								We do not use your audio recordings to train our machine
								learning models without your explicit consent.
							</li>
						</ul>
					</Section>

					<Section title="4. Third-Party Services">
						<p>We use the following third-party services:</p>
						<ul className="list-disc pl-6 mt-3 space-y-2">
							<li>
								<strong className="text-cream">Apple:</strong> Authentication
								(Sign in with Apple)
							</li>
							<li>
								<strong className="text-cream">Cloudflare:</strong> Hosting,
								content delivery, and data storage
							</li>
							<li>
								<strong className="text-cream">AI Model Providers:</strong> We
								use third-party large language models to generate practice
								feedback. Practice context (not raw audio) may be shared with
								these providers subject to their data processing agreements.
							</li>
							<li>
								<strong className="text-cream">Sentry:</strong> Error tracking
								and performance monitoring
							</li>
						</ul>
					</Section>

					<Section title="5. Cookies and Local Storage">
						<p>
							We use essential cookies for authentication (session tokens) and
							local storage for application state. We do not use advertising or
							tracking cookies.
						</p>
					</Section>

					<Section title="6. Data Retention">
						<p>
							We retain your account and practice data for as long as your
							account is active. Audio recordings are processed in real time and
							not retained after analysis. If you delete your account, we will
							delete your personal data within 30 days, except where retention
							is required by law.
						</p>
					</Section>

					<Section title="7. Data Security">
						<p>
							We implement industry-standard security measures to protect your
							data, including encryption in transit (TLS) and at rest. However,
							no method of transmission or storage is completely secure, and we
							cannot guarantee absolute security.
						</p>
					</Section>

					<Section title="8. Your Rights">
						<p>Depending on your jurisdiction, you may have the right to:</p>
						<ul className="list-disc pl-6 mt-3 space-y-2">
							<li>Access the personal data we hold about you</li>
							<li>Request correction of inaccurate data</li>
							<li>Request deletion of your data</li>
							<li>Export your data in a portable format</li>
							<li>Opt out of certain data processing activities</li>
						</ul>
						<p className="mt-3">
							To exercise these rights, contact us at the address below.
						</p>
					</Section>

					<Section title="9. Children's Privacy">
						<p>
							The Service is not directed at children under 13. We do not
							knowingly collect personal information from children under 13. If
							you believe a child under 13 has provided us with personal
							information, please contact us and we will delete it.
						</p>
					</Section>

					<Section title="10. Changes to This Policy">
						<p>
							We may update this Privacy Policy from time to time. We will
							notify you of material changes by posting the updated policy on
							the Service with a new effective date. Continued use of the
							Service after changes constitutes acceptance of the revised
							policy.
						</p>
					</Section>

					<Section title="11. Contact">
						<p>
							If you have questions about this Privacy Policy or our data
							practices, please contact us through the Service.
						</p>
					</Section>
				</div>
			</div>
		</div>
	);
}

function Section({
	title,
	children,
}: {
	title: string;
	children: React.ReactNode;
}) {
	return (
		<section>
			<h2 className="font-display text-display-sm text-cream mb-4">
				{title}
			</h2>
			{children}
		</section>
	);
}
