import { and, eq, sql } from "drizzle-orm";
import { segmentLoops } from "../db/schema/segment-loops";
import type { SEGMENT_LOOP_STATUSES } from "../db/schema/segment-loops";
import { ConflictError, NotFoundError, ValidationError } from "../lib/errors";
import { DIMS_6 } from "../lib/dims";
import type { Db } from "../lib/types";
import type { SegmentLoopArtifact } from "../harness/artifacts/segment-loop";
import type { InlineComponent } from "./tool-processor";

type SegmentLoopStatus = (typeof SEGMENT_LOOP_STATUSES)[number];

const TERMINAL_STATUSES: SegmentLoopStatus[] = ["completed", "dismissed", "superseded"];

function rowToArtifact(row: typeof segmentLoops.$inferSelect): SegmentLoopArtifact {
  if (!SEGMENT_LOOP_STATUSES.includes(row.status as SegmentLoopStatus)) {
    throw new ValidationError(`invalid segment_loop status in DB: '${row.status}'`);
  }
  if (row.dimension !== null && !(DIMS_6 as readonly string[]).includes(row.dimension)) {
    throw new ValidationError(`invalid segment_loop dimension in DB: '${row.dimension}'`);
  }
  return {
    kind: "segment_loop",
    id: row.id,
    studentId: row.studentId,
    pieceId: row.pieceId,
    barsStart: row.barsStart,
    barsEnd: row.barsEnd,
    requiredCorrect: row.requiredCorrect,
    attemptsCompleted: row.attemptsCompleted,
    status: row.status as SegmentLoopStatus,
    dimension: row.dimension as SegmentLoopArtifact["dimension"],
  };
}

export interface CreateSegmentLoopInput {
  studentId: string;
  pieceId: string;
  conversationId: string | null;
  barsStart: number;
  barsEnd: number;
  requiredCorrect: number;
  dimension?: string | null;
  trigger: "chat" | "synthesis";
}

export async function createSegmentLoop(
  db: Db,
  input: CreateSegmentLoopInput,
): Promise<SegmentLoopArtifact> {
  const status = input.trigger === "chat" ? "pending" : "active";

  return await db.transaction(async (tx) => {
    // Supersede any non-terminal loop for this (student, piece)
    await tx
      .update(segmentLoops)
      .set({ status: "superseded", updatedAt: new Date() })
      .where(
        and(
          eq(segmentLoops.studentId, input.studentId),
          eq(segmentLoops.pieceId, input.pieceId),
          sql`status NOT IN ('completed', 'dismissed', 'superseded')`,
        ),
      );

    const [inserted] = await tx
      .insert(segmentLoops)
      .values({
        studentId: input.studentId,
        pieceId: input.pieceId,
        conversationId: input.conversationId,
        barsStart: input.barsStart,
        barsEnd: input.barsEnd,
        requiredCorrect: input.requiredCorrect,
        dimension: input.dimension ?? null,
        trigger: input.trigger,
        status,
      })
      .returning();

    if (!inserted) throw new ConflictError("Failed to insert segment loop");
    return rowToArtifact(inserted);
  });
}

async function assertOwnerAndLoad(
  db: Db,
  id: string,
  studentId: string,
): Promise<typeof segmentLoops.$inferSelect> {
  const [row] = await db
    .select()
    .from(segmentLoops)
    .where(eq(segmentLoops.id, id))
    .limit(1);
  if (!row || row.studentId !== studentId) throw new NotFoundError("segment_loop", id);
  return row;
}

function assertNotTerminal(status: string, id: string): void {
  if (TERMINAL_STATUSES.includes(status as SegmentLoopStatus)) {
    throw new ValidationError(`segment_loop ${id} is in terminal state '${status}'`);
  }
}

export async function acceptSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "pending") {
    throw new ValidationError(`accept requires status 'pending', got '${row.status}'`);
  }
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "active", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function declineSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "pending") {
    throw new ValidationError(`decline requires status 'pending', got '${row.status}'`);
  }
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "dismissed", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function dismissSegmentLoop(
  db: Db,
  id: string,
  studentId: string,
): Promise<SegmentLoopArtifact> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  assertNotTerminal(row.status, id);
  const [updated] = await db
    .update(segmentLoops)
    .set({ status: "dismissed", updatedAt: new Date() })
    .where(eq(segmentLoops.id, id))
    .returning();
  return rowToArtifact(updated!);
}

export async function findActiveForPiece(
  db: Db,
  studentId: string,
  pieceId: string,
): Promise<SegmentLoopArtifact | null> {
  const [row] = await db
    .select()
    .from(segmentLoops)
    .where(
      and(
        eq(segmentLoops.studentId, studentId),
        eq(segmentLoops.pieceId, pieceId),
        eq(segmentLoops.status, "active"),
      ),
    )
    .limit(1);
  return row ? rowToArtifact(row) : null;
}

export async function incrementAttempts(
  db: Db,
  id: string,
  studentId: string,
): Promise<{ attemptsCompleted: number; completedNow: boolean }> {
  const row = await assertOwnerAndLoad(db, id, studentId);
  if (row.status !== "active") {
    throw new ValidationError(`incrementAttempts requires status 'active', got '${row.status}'`);
  }
  const newCount = row.attemptsCompleted + 1;
  const completedNow = newCount >= row.requiredCorrect;
  await db
    .update(segmentLoops)
    .set({
      attemptsCompleted: newCount,
      status: completedNow ? "completed" : "active",
      updatedAt: new Date(),
    })
    .where(eq(segmentLoops.id, id));
  return { attemptsCompleted: newCount, completedNow };
}

export function toLoopComponent(artifact: SegmentLoopArtifact): InlineComponent {
  return {
    type: "segment_loop",
    config: {
      id: artifact.id,
      pieceId: artifact.pieceId,
      barsStart: artifact.barsStart,
      barsEnd: artifact.barsEnd,
      requiredCorrect: artifact.requiredCorrect,
      attemptsCompleted: artifact.attemptsCompleted,
      status: artifact.status,
      dimension: artifact.dimension,
    },
  };
}
