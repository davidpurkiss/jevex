Extract structured records from the document.

The record types and their fields:

{schemas}

Return every record of these types that the document describes, one per distinct item
(for example one per vehicle variant in a specification, or one per car for sale), in a
list under its type's name. Leave a type's list empty when the document describes none.

For each field, give the value the document states for that record. Give numbers in the
field's unit, converting them if the document uses another unit. Use null for a field the
document doesn't state. Don't guess or fill in values from general knowledge.
